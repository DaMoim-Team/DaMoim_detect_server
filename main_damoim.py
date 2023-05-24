
# model load용
import cv2
import torch
from ultralytics import YOLO

# DB연결
from database import update_database, update_ranktable

# count 랭킹용
from count_total import Count_total
from oneday_rank import Oneday
from time_range import current_range
import schedule
import time

# Detection 설정
import os
import glob
from track_crop_damoim import track_crop  # tracking and crop
from openpose_damoim import arm_openpose  # 팔각도 openpose
from yolo_damoim import smoke_head_yolo  # 흡연자머리 Yolov8
from config import Config  # 구성값 설정파일
import shutil

count = Count_total()  # 전역변수로 첫 초기화
oneday = Oneday()  # 전역변수로 첫 초기화 

# 한시간 싸이클
def h_cycle():
    global count
    top_cctv = max(count.count_dict, key=count.count_dict.get)
    top_count = count.count_dict[top_cctv]
    cur_range = current_range()
    oneday.oneday_dict[cur_range] = [top_cctv, top_count]
    count = Count_total() # 한시간 주기 초기화

# 하루 싸이클
def d_cycle():
    update_ranktable(oneday.oneday_dict)
    
# 한시간 단위 스케줄링
for i in range(10, 19):
    schedule.every().day.at(f"{i:02d}:00").do(h_cycle)

# 하루 단위 스케줄링
schedule.every().day.at("18:10").do(d_cycle)

# Detect Path & 검출완료 디렉토리 삭제함수 세팅
input_base_path = Config.INPUT_BASE_PATH
output_base_path = Config.OUTPUT_BASE_PATH

def remove(path):
    for root_dir, dirs, _ in os.walk(path, topdown=False):
        for dir in dirs:
            shutil.rmtree(os.path.join(root_dir, dir))

### openpose model load ###
net = cv2.dnn.readNetFromCaffe(Config.PROTO_FILE, Config.WEIGHTS_FILE)

if Config.DEVICE_OP == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif Config.DEVICE_OP == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

### SORT model load ###
device_sort = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_sort = torch.hub.load(Config.MODEL_SORT, Config.MODEL_SORT_VERSION, device = device_sort)

### Yolov8 model load ###
model_yolo = YOLO(Config.MODEL_YOLO)

# 'complete.txt' 파일의 경로
complete_file_path = Config.COMPLETE_PATH
check = False

### 영상 수신시간 기준으로 반복한다 ###

while(True):
    schedule.run_pending() # 무한루프 내에서 1시간, 하루 간격 스케쥴링
    time.sleep(0.5) # 자원 과다사용 방지

    # complete.txt 내용 확인
    if os.path.exists(complete_file_path):
        with open(complete_file_path, 'r') as f:
            content = f.read().strip()
            if content == 'complete':
                check = True
                    
    while(check): # 라즈베리 파이에서 영상 수신완료 받았을 경우에 실행될 루프 (complete.txt 체크 코드)
        print("영상 수신 완료")
        # 모든 mp4 파일 검색
        video_files = glob.glob(os.path.join(input_base_path, '**', '*.mp4'), recursive=True)

        # 각 topic별로 검출수 저장하는 딕셔너리 생성
        detect_values = {}

        for video_file in video_files:
            input_video_path = video_file
            relative_path = os.path.relpath(video_file, input_base_path)
            folder_structure = os.path.split(os.path.splitext(relative_path)[0])[0]
            cctv_name = folder_structure.split(os.sep)[0]  # cctv_1, cctv_2, ...
            output_folder_path = os.path.join(output_base_path, folder_structure)

            # 해당 output 폴더가 없으면 생성
            os.makedirs(output_folder_path, exist_ok=True)

                                         ### SORT ###

            # (model ,crop할 directory, 저장할 directory)
            track_crop(model_sort, input_video_path, output_folder_path)
            
                                        ### opnepose ###

            # (openpose 적용할 directory / mdeol network / low_threshold / high_threshold / 검출 기준점 point)
            detect_value_op = arm_openpose(output_folder_path, net, 50, 60, 5)

                                      ### custom Yolov8 ###

            # (model, yolo 적용할 directory, 검출 기준점 point)
            detect_value_yo = smoke_head_yolo(model_yolo ,output_folder_path, 11)

            
                                    ### openpose + yolo 앙상블 ###

            # openpose, yolo 둘다 1을 받아온 경우, 합해져 2
            result_sum = [a + b for a, b in zip(detect_value_op, detect_value_yo)]
            # 2이상 값만 1로 변환하고 그 미만은 0으로 통일한다. 그 후 1값 count
            result = [1 if value >= 2 else 0 for value in result_sum].count(1)

            # cctv key에 맞게 value 대입
            detect_values[cctv_name] = result

                                  ### 최종 결과 detect_values 출력 ###

        for cctv, value in detect_values.items():
            print(f"{cctv}: {value}")

        # DB update
        update_database(detect_values)

        # count 갱신
        for key in count.count_dict.keys() & detect_values.keys():
            count.count_dict[key] += detect_values[key]
                
        # results 하위의 검출 끝낸 frames 삭제
        remove(output_base_path)

        # cctvs 하위의 검출 끝낸 mp4 삭제
        remove(input_base_path)        

        if os.path.exists(complete_file_path):
            # complete.txt 내용 초기화
            with open(complete_file_path, 'w') as file:
                pass
        
        # 한 주기의 끝을 선언
        check = False
        print("\n...대기모드...\n")


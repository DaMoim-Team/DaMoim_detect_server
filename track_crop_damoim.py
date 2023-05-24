import os
import time
import cv2
import sys
import torch
from sort_damoim import Sort

def track_crop(model_damoim, input_video_path, output_folder_path):

    # 이전수행 track id 기록용 함수 
    def get_max_track_id(output_folder):
        max_track_id = -1
        for folder_name in os.listdir(output_folder):
            if folder_name.startswith('person_'):
                track_id = int(folder_name.split('_')[1])
                if track_id > max_track_id:
                    max_track_id = track_id
        return max_track_id
    
    # YOLOv5 모델 로드
    model = model_damoim

    # SORT 객체 생성
    mot_tracker = Sort()

    # 입력 비디오 로드
    cap = cv2.VideoCapture(input_video_path)

    # 출력 폴더 설정
    output_folder = output_folder_path

    # 기존의 최대 track_id 값을 찾기
    max_existing_track_id = get_max_track_id(output_folder)

    # 4 주기로 frame 저장
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 4배수 아닐땐 skip
        if frame_counter % 4 != 0:
            frame_counter += 1
            continue

        # YOLOv5로 객체 탐지
        results = model(frame)

        # 사람 객체만 추출
        person_detections = results.xyxy[0][results.xyxy[0][:, -1] == 0]

        # SORT 추적 수행
        tracked_objects = mot_tracker.update(person_detections)

        padding = 20  # 원하는 패딩 값을 설정

        for track in tracked_objects:
            bbox = track[:4].astype(int)
            track_id = int(track[4]) + max_existing_track_id + 1  # 기존의 최대 track_id보다 큰 값부터 시작
            x_min, y_min, x_max, y_max = bbox

            width = x_max - x_min
            height = y_max - y_min

            # bounding box가 너무 작은 person은 무시 (멀리 지나가는 사람까지 crop하는것 방지)
            if width < 40 or height < 50:
                continue            
            
            # Bounding box 좌표에 패딩을 추가
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)

            # Crop the person from the frame
            person_img = frame[y_min:y_max, x_min:x_max]

            # Create a new folder for this person using the track_id
            person_folder = os.path.join(output_folder, f"person_{track_id}")
            os.makedirs(person_folder, exist_ok=True)

            # Save the person image to the new folder
            current_time = time.time()
            formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            milliseconds = int((current_time - int(current_time)) * 1000)
            cv2.imwrite(os.path.join(person_folder, f"{formatted_time}_{milliseconds}_person_{track_id}.jpg"), person_img)
        
        # frame_counter 증가
        frame_counter += 1

    cap.release()

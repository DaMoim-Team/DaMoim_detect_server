#동영상 받아와서 저장하기

import time
import paho.mqtt.client as mqtt
import queue
import os
import base64
import datetime
import pytz
import cv2
from pathlib import Path
from config import Config  # 구성값 설정파일

#브로커 IP
broker_address = Config.BROKER_ADDRESS

#한국으로 timezone 설정
korea_timezone = pytz.timezone('Asia/Seoul')
current_time = datetime.datetime.now(korea_timezone)

isRunning = True
myqueue = queue.Queue(1)
video_count = 0

def onStart():
	client.publish("command", "start")

def onStop():
	client.publish("command", "stop")

def onExit():
	global isRunning
	client.publish("command", "stop")
	isRunning = False

#브로커 성공적 연결
def onConnect(client, userdata, flag, rc):
    if(rc == 0):
        print("연결되었습니다")
        #토픽 수신 대기
        client.subscribe("cctv_1", qos = 0)
        client.subscribe("cctv_2", qos = 0)
        client.subscribe("cctv_3", qos = 0)
        client.subscribe("cctv_4", qos = 0)
        client.subscribe("damoim", qos = 0)
    else:
        print("연결 실패: ", rc)

video_data = b''
video_output = 'output.mp4'
start_time = time.time()

#청크를 모두 전달받았다는 신호
video_boundary = b'--video-boundary--'
#모든 토픽의 영상을 수신 완료했다는 신호
complete = b'complete'
#더미 데이터 수신
dummy = b'dummy'

is_video_data = False

start_time = None

#메시지 도착시 호출
def onMessage(client, userdata, msg):
    global video_data, video_output, is_video_data, start_time
    
    ascii_chunk = msg.payload
    #받은 메시지 base64로 디코딩
    decoded_chunk = base64.b64decode(ascii_chunk)


    # 메시지가 'complete' 일때
    if decoded_chunk == complete:
        print("----all cctv received complete----")
        # complete.txt 파일 경로
        complete_file_path = Config.COMPLETE_PATH
        
        # complete.txt 파일 작성 (덮어쓰기)
        with open(complete_file_path, 'w') as f:
            f.write('complete')
        print("complete.txt 파일 업데이트")
        
    elif decoded_chunk == video_boundary:
        if is_video_data and len(video_data) > 0:
            #영상 끝
            topic_folder = msg.topic

            #.mp4형식으로 영상 파일 저장
            formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
            video_output = f'{Config.INPUT_BASE_PATH}/{topic_folder}/{topic_folder}_{formatted_time}.mp4'
            os.makedirs(os.path.dirname(video_output), exist_ok=True)

            with open(video_output, 'wb') as video_file:
                video_file.write(video_data)
            print(f"Video saved as {video_output}")
            
            #다시 수행하기 위해 변수 초기화
            video_data = b''
            
            #메시지 수신 후
            end_time = time.time()
            # 처리 시간 계산
            processing_time = end_time - start_time
            print(f"Processing time: {processing_time} seconds")
        else:
            #영상 시작
            is_video_data = True
            start_time = time.time()  #영상 시작
    elif is_video_data:
        video_data += decoded_chunk


# MQTT 제어
client = mqtt.Client()		# mqtt 클라이언트 객체 생성
client.on_connect = onConnect	# 연결요청시 Callback 함수
client.on_message = onMessage	# 이미지가 도착하였을때 Callback 함수
client.connect(broker_address, 1883)	# 브로커에 연결을 요청함
client.loop_start()  #비동기

#브로커에서 메시지 계속 수신하도록 유지
while(isRunning):
    time.sleep(0.5)

client.loop_end()
client.disconnect()

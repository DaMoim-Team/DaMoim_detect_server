

## 📁 디렉토리 구조
```
📦detect_server
 ┣ 📂cctvs                            - 샘플 input directory
 ┃ ┣ 📂cctv_1
 ┃ ┣ 📂cctv_2
 ┃ ┣ 📂cctv_3
 ┃ ┗ 📜complete.txt                   - 영상수신완료 식별용  
 ┃
 ┣ 📂docs                             - 참고문서
 ┃
 ┣ 📂models                           - model 모음
 ┃ ┣ 📜pose_deploy.prototxt
 ┃ ┣ 📜pose_iter_584000.caffemodel
 ┃ ┣ 📜yolov5s.pt
 ┃ ┗ 📜yolov8_smoking.pt
 ┣ 📂results                          - 샘플 output directory
 ┃ ┣ 📂cctv_1
 ┃ ┣ 📂cctv_2
 ┃ ┗ 📂cctv_3
 ┣ 📜count_total.py                   - 시간단위 검출데이터 class
 ┣ 📜database.py                      - DB 전송
 ┣ 📜GPU_setting_check.py             - GPU 세팅체크
 ┣ 📜main_damoim.py                   - main 구동
 ┣ 📜oneday_rank.py                   - 하루단위 검출데이터 class
 ┣ 📜openpose_damoim.py               - openpose
 ┣ 📜sort_damoim.py                   - SORT
 ┣ 📜subscribe.py                     - 영상 받기
 ┣ 📜time_range.py                    - 시간구간 체크
 ┣ 📜track_crop_damoim.py             - TRACKING + CROP
 ┣ 📜YOLOv8_train.ipynb               - YOLOv8 전이학습 코드
 ┗ 📜yolo_damoim.py                   - YOLOv8
```

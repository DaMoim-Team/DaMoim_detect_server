import os

def smoke_head_yolo(model_damoim, rootdir_damoim, point_damoim):

    # yolov8 흡연자 판단 알고리즘
    def Decide_smoker(smoker_head_list, point):
        count = smoker_head_list.count(1)
        if count >= point:
            return 1
        else:
            return 0

    # custom Yolov8 모델 설정
    model_yolo = model_damoim

    root_dir = rootdir_damoim  #  main_damoim에서 인자로 받아오기

    smoker_head_detect = []

    for person_id in sorted(os.listdir(root_dir), key=lambda x: int(x.split('_')[1])):
        person_dir = os.path.join(root_dir, person_id)
        if os.path.isdir(person_dir):
            detected_boxes = []

            for file in sorted(os.listdir(person_dir), key=lambda x: float(x.split('_')[0])):
                filepath = os.path.join(person_dir, file)
                                
                if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Detection 시작
                    results = model_yolo.predict(
                        conf=0.7,
                        source=filepath,
                    )
                    detected_boxes.append(len(results[0].boxes))
                
        result_per = Decide_smoker(detected_boxes, point_damoim)
        smoker_head_detect.append(result_per)

    # 최종 작업 return
    return smoker_head_detect # list로 반환 (person_id 순서로 index 구성)

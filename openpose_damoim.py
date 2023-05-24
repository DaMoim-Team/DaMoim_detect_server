import os
import cv2
import time
import numpy as np
import math

def arm_openpose(rootdir_damoim, net_damoim, low_threshold_damoim, high_threshold_damoim, point_damoim):
    root_dir = rootdir_damoim  #  main_damoim에서 인자로 받아오기

    nPoints = 8
    keypointsMapping = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]
    POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7]]
    mapIdx = [[40,41], [48, 49], [42, 43], [44, 45], [50, 51], [52, 53]]
    
    def euclidean_distance(pt1, pt2):
        return np.sqrt(np.sum((pt1 - pt2) ** 2))

    def angle_between_three_points(a, b, c):
        ab = np.array(a) - np.array(b)
        cb = np.array(c) - np.array(b)
        
        dot_product = np.dot(ab, cb)
        ab_length = np.linalg.norm(ab)
        cb_length = np.linalg.norm(cb)
        
        cos_angle = dot_product / (ab_length * cb_length)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle

    def getKeypoints(probMap, threshold=0.1):
        
        mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

        mapMask = np.uint8(mapSmooth>threshold)
        keypoints = []
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
        return keypoints

    def getValidPairs(output):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        for k in range(len(mapIdx)):
            pafA = output[0, mapIdx[k][0], :, :]
            pafB = output[0, mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (frameWidth, frameHeight))
            pafB = cv2.resize(pafB, (frameWidth, frameHeight))

            candA = detected_keypoints[POSE_PAIRS[k][0]]
            candB = detected_keypoints[POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)
            
            if( nA != 0 and nB != 0):
                valid_pair = np.zeros((0,3))
                for i in range(nA):
                    max_j=-1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)
                        
                        if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    if found:            
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)
                        
                valid_pairs.append(valid_pair)
            else:
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs

    def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
        personwiseKeypoints = -1 * np.ones((0, 26))

        for k in range(len(mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])): 
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                    elif not found and k < 24:
                        row = -1 * np.ones(26)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints    

    def calculate_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    # openpose 흡연자 판단 알고리즘
    def find_angle_transition_pattern(angleslist, low_threshold, high_threshold, point):
        smoker = False
        count = 0
        for i in range(1, len(angleslist)):
            if angleslist[i - 1] >= high_threshold and angleslist[i] <= low_threshold:
                count += 1
        if count >= point:
            smoker = True
        else:
            smoker = False
        
        return smoker    

    ######################## OpenPose모델 적용 시작 ###########################
    
    net = net_damoim # model load
    low_threshold = low_threshold_damoim  # 다음 frame
    high_threshold = high_threshold_damoim # 이전 frame
    point = point_damoim   # 흡연패턴 검출 기준점

    smoker_detect = []   # root directrory에서의 흡연자 검출수 리스트

    for person_id in sorted(os.listdir(root_dir), key=lambda x: int(x.split('_')[1])): 
        person_dir = os.path.join(root_dir, person_id)
        if os.path.isdir(person_dir):
            right_angle_list = []
            left_angle_list = []

            for file in sorted(os.listdir(person_dir), key=lambda x: float(x.split('_')[0])):
                filepath = os.path.join(person_dir, file)

                # 다른 포인트 검출에 영향을 받지 않기 위해 flag변수 설정
                flag_r = 0
                flag_l = 0

                if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image1 = cv2.imread(filepath)
                    
                    #  1. 이미지 크기를 절반으로 수행시간 단축
                    image1 = cv2.resize(image1, (image1.shape[1] // 2, image1.shape[0] // 2))
                    
                    frameWidth = image1.shape[1]
                    frameHeight = image1.shape[0]

                    #  2. 이미지 비율 조정으로 수행시간 단축
                    inHeight = 123
                    inWidth = int((inHeight/frameHeight)*frameWidth)

                    inpBlob = cv2.dnn.blobFromImage(image1,
                                                1.0 / 255, (inWidth, inHeight),
                                            (0, 0, 0), swapRB=False, crop=False)

                    
                    net.setInput(inpBlob)
                    output = net.forward()

                    detected_keypoints = []
                    keypoints_list = np.zeros((0,3))
                    keypoint_id = 0
                    threshold = 0.1

                    for part in range(nPoints):
                        probMap = output[0,part,:,:]
                        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
                        keypoints = getKeypoints(probMap, threshold)
                        keypoints_with_id = []
                        
                        
                        for i in range(len(keypoints)):
                            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                            keypoint_id += 1

                        detected_keypoints.append(keypoints_with_id)

                    frameClone = image1.copy()
                    for i in range(nPoints):
                        for j in range(len(detected_keypoints[i])):
                            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, (0, 255, 0), -1, cv2.LINE_AA)

                        valid_pairs, invalid_pairs = getValidPairs(output)
                        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

                    for i in range(len(POSE_PAIRS)):
                        for n in range(len(personwiseKeypoints)):
                        
                            # 오른팔 각도
                            right_shoulder = keypoints_list[int(personwiseKeypoints[n][2])][:2]
                            right_elbow = keypoints_list[int(personwiseKeypoints[n][3])][:2]
                            right_wrist = keypoints_list[int(personwiseKeypoints[n][4])][:2]
                            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                            # 왼팔 각도
                            left_shoulder = keypoints_list[int(personwiseKeypoints[n][5])][:2]
                            left_elbow = keypoints_list[int(personwiseKeypoints[n][6])][:2]
                            left_wrist = keypoints_list[int(personwiseKeypoints[n][7])][:2]
                            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                            if flag_l < 1:
                                left_angle_list.append(left_arm_angle.item())
                                flag_l += 1
                            if flag_r < 1:
                                right_angle_list.append(right_arm_angle.item())
                                flag_r += 1

        # 흡연자 패턴 검출 알고리즘 적용
        left_smoker = find_angle_transition_pattern(left_angle_list, low_threshold, high_threshold, point)
        right_smoker = find_angle_transition_pattern(right_angle_list, low_threshold, high_threshold, point)

        # 왼팔 or 오른팔 중 패턴 검출시 흡연자 판정        
        if left_smoker or right_smoker:
            smoker_detect.append(1)
        else:
            smoker_detect.append(0)
    
    return smoker_detect   # list로 반환 (person_id 순서로 index 구성)



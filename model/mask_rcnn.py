import os
import sys
import random
import math
import numpy as np
import skimage.io

from model.mrcnn import utils
import model.mrcnn.model as modellib
from model.mrcnn import visualize
import cv2
from PIL import Image
from model.coco import coco2
import warnings

import json
import io
import requests

class BackgroundMasking:
    def load_model(self):
        # 경로 설정
        ROOT_DIR = os.path.abspath("./") #현재 경로
        sys.path.append(ROOT_DIR)

        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
        
        # 모델 불러오기    
        class InferenceConfig(coco2.CocoConfig): # 'coco.py'의 'CocoConfig' 클래스
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            
        config = InferenceConfig()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        return model
    
    def background_masking(self, model, input, NUM): # 
        out_list = []
        # COCO Class names
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']
        
        # input 파일에 들어있는 모든 jpg 사진 처리
        file_list_py = [file for file in input if file.endswith(('.jpg', '.png', '.jpeg'))]
        print('file_list_py', file_list_py)
        ## 이미지 블러 처리 인풋 폴더 안 사진 파일 file_list_py 돌면서 실행됨
        # model = self.load_model()
        num = 0

        for filename in file_list_py:
            
            image = skimage.io.imread(os.path.join(filename))
            # png 처리 위한 코드
            if image.shape[2] == 4:
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image = np.array(image)
            
            results= model.detect([image],verbose=0)
            
            r = results[0]
            
            # 임계값 설정
            threshold = 0.9

            indices_to_keep = np.where(r['scores'] > threshold)[0]

            # 선택한 인덱스에 해당하는 masks, scores, class_ids를 가져옵니다.
            r['masks'] = r['masks'][:, :, indices_to_keep]
            r['scores'] = r['scores'][indices_to_keep]
            r['class_ids'] = r['class_ids'][indices_to_keep]

            masks = r['masks'][:, :, r['class_ids'] == 1]  # 사람인 경우
            
            if not masks.size > 0:
                print('NO person!!')
                out_list.append(filename)
            else:
                mask = np.sum(masks, axis=2).astype(np.bool)  # 채널 하나짜리 마스크
                # 확장된 마스크 생성
                kernel = np.ones((25, 25), np.uint8)  # 팽창 연산에 사용될 커널 설정
                expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
                expanded_mask = expanded_mask.astype(np.bool)
                mask_3d = np.repeat(np.expand_dims(expanded_mask, axis=2), 3, axis=2).astype(np.uint8)  # 채널 3짜리 마스크

                # 이미지 블러 처리
                blurred_img = cv2.GaussianBlur(image, (101, 101), 125)
                mask_3d_blurred = (cv2.GaussianBlur(mask_3d*255,(101,101),10,10)/255).astype(np.float32)

                # mix it together
                person_mask = mask_3d_blurred * image.astype(np.float32)
                bg_mask = (1 - mask_3d_blurred) * blurred_img.astype(np.float32)
                out = (person_mask + bg_mask).astype(np.uint8)
                out_image = out.astype(np.uint8)
                
                bgr_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR) # 최종본!
                print(filename, 'finish!') # 잘되는지 안되는지 확인
                
                #img_name = 'img'+ str(NUM) + '.jpg'
                #output_path = os.path.join(input, img_name) #위에서 정의한 경로에 filname으로 저장하기 위한 코드
                output_path = filename
                cv2.imwrite(output_path, bgr_image)# 아웃풋 폴
            
                out_list.append(output_path)
                print('output_path',output_path)
                NUM += 1
        
        return out_list, NUM
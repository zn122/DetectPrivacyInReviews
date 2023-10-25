from __future__ import division
import torch
import os
import cv2
import numpy as np
from face_model.common.utils import BBox, drawLandmark, drawLandmark_multiple
from face_model.models.basenet import MobileNet_GDConv
from face_model.blur.bluring import total_blur
from PIL import Image
from face_model.MTCNN import detect_faces
from face_model.utils.align_trans import get_reference_facial_points, warp_and_crop_face

import json
import io
import requests


# 이미지 파일 경로 설정
script_directory = os.path.dirname(os.path.abspath(__file__))

class FaceMasking:
    map_location = 0
    mean = 0
    std =0
    
    def __init__(self):
        self.initialize()
        
    def initialize(self, ):
        FaceMasking.mean = np.asarray([0.485, 0.456, 0.406])
        FaceMasking.std = np.asarray([0.229, 0.224, 0.225])

        crop_size = 112
        scale = crop_size / 112.
        reference = get_reference_facial_points(default_square=True) * scale

        if torch.cuda.is_available():
            FaceMasking.map_location = lambda storage, loc: storage.cuda()
        else:
            FaceMasking.map_location = 'cpu'
            

    def load_model(self):
        model = MobileNet_GDConv(136)
        model = torch.nn.DataParallel(model)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_directory, 'checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar')
        checkpoint = torch.load(checkpoint_path, map_location=FaceMasking.map_location)
        print(FaceMasking.map_location)
        print('Use MobileNet as backbone')
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def face_masking(self, model, input): #
        output_folder = 'output'
        out_size = 224
        # model = self.load_model()
        model = model.eval()

        # input 파일에 들어있는 모든 jpg 사진 처리
        file_list_py = [file for file in input if file.endswith(('.jpg', '.png', '.jpeg'))]
        print(file_list_py)
        # 이미지 파일 순회 및 처리
        for file_name in file_list_py:
            print(file_name)  
            img = cv2.imread(file_name)
            height, width, _ = img.shape
            ##detector
            image = Image.open(file_name)
            faces, landmarks = detect_faces(image)
            # png 파일
            if image.mode == "RGBA":
                image = image.convert("RGB")
            # C:\Users\hOMe_pc\Code\AIs\server\FlaskTest\face_model\MTCNN\detector.py 수정
            
            result_img = []
            print('landmarks',landmarks)
            ratio = 0
            if len(landmarks) == 0:
                print('NO face is detected!')
                
            if len(faces) == 0:
                print('NO face is detected!')
                
            else:
                # Initialize an empty list to store landmarks
                all_landmarks = []

                for k, face in enumerate(faces):
                    if face[4] < 0.9:  # remove low confidence detection
                        continue
                    x1 = face[0]
                    y1 = face[1]
                    x2 = face[2]
                    y2 = face[3]
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1
                    size = int(min([w, h]) * 1.2)
                    cx = x1 + w // 2
                    cy = y1 + h // 2
                    x1 = cx - size // 2
                    x2 = x1 + size
                    y1 = cy - size // 2
                    y2 = y1 + size

                    dx = max(0, -x1)
                    dy = max(0, -y1)
                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    edx = max(0, x2 - width)
                    edy = max(0, y2 - height)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    new_bbox = list(map(int, [x1, x2, y1, y2]))
                    new_bbox = BBox(new_bbox)
                    cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
                    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
                    cropped_face = cv2.resize(cropped, (out_size, out_size))

                    if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                        continue
                    test_face = cropped_face.copy()
                    test_face = test_face / 255.0
                    test_face = (test_face - FaceMasking.mean) / FaceMasking.std
                    test_face = test_face.transpose((2, 0, 1))
                    test_face = test_face.reshape((1,) + test_face.shape)
                    input = torch.from_numpy(test_face).float()
                    input = torch.autograd.Variable(input)
                    
                    landmark = model(input).cpu().data.numpy()
                    
                    landmark = landmark.reshape(-1, 2)
                    landmark = new_bbox.reprojectLandmark(landmark)
                    
                    # Append the landmarks to the list
                    all_landmarks.append(landmark)

                # Draw landmarks on the original image
                for landmarks in all_landmarks:
                    
                    img = total_blur(img, landmarks)
                
                # img_name = 'img'+ str(NUM) + '.jpg'
                # output_path = os.path.join(output_folder, img_name) #위에서 정의한 경로에 filname으로 저장하기 위한 코드
                cv2.imwrite(file_name, img)# 아웃풋 저장 덮어씌우기
        
        return 
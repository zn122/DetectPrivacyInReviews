import numpy as np
import cv2

def blur_mouse(img, landmarks):
    

    # 얼굴 크기에 맞게 영역 크기 조절
    face_width = landmarks[16][0] - landmarks[0][0]  # 얼굴의 가로 너비 계산
    face_height = landmarks[8][1] - landmarks[27][1]  # 얼굴의 세로 높이 계산
    
    mouth_indices = list(range(48, 61))
    
    mouth_points = [(landmarks[index][0], landmarks[index][1]) for index in mouth_indices]

    # 입 주변 영역 크기 조절
    for i in range(1, 6):
        mouth_points[i] = (mouth_points[i][0], mouth_points[i][1] - int(face_height * 0.05))
    mouth_points[0] = (mouth_points[0][0] - int(face_width * 0.05), mouth_points[0][1])
    mouth_points[6] = (mouth_points[6][0] + int(face_width * 0.05), mouth_points[6][1])
    mouth_points[12] = (mouth_points[12][0] - int(face_width * 0.05), mouth_points[12][1])
    for i in range(7, 13):
        mouth_points[i] = (mouth_points[i][0], mouth_points[i][1] + int(face_height * 0.05))

    # 다각형 생성 및 마스킹
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], 255)
    # 마스크를 사용하여 블러 처리
    blurred_mouth_region = cv2.GaussianBlur(img, (0, 0), sigmaX=30, sigmaY=30, borderType=cv2.BORDER_DEFAULT)

    # 블러 처리된 영역을 원본 이미지에 적용
    img = np.where(mask[:, :, None] == 255, blurred_mouth_region, img)

#     # 결과 이미지 표시
#     cv2.imshow("lsh", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return img

def blur_nose(img, landmarks):
    # 코 완성
    face_height = landmarks[8][1] - landmarks[27][1]  # 얼굴의 세로 너비 계산
    face_width = landmarks[16][0] - landmarks[0][0]  # 얼굴의 가로 너비 계산
    ## NOSE = list(range(27, 36))
    nose_indices = [27, 35, 34, 33,32, 28, 31]
    
    
    nose_offsets = [(0, 0), (int(0.1 * face_width), 0), (int(0.05 * face_width), 0), (int(0.05 * face_width), 0),
                    (-int(0.1 * face_width), 0), (-int(0.05 * face_width), 0)]

    # 코 주변 영역의 좌표 계산 및 크기 조정
    polygon_points = [(landmarks[index][0] + offset[0], landmarks[index][1] + offset[1]) for index, offset in zip(nose_indices, nose_offsets)]

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_points, dtype=np.int32)], 255)

    # 마스크를 사용하여 블러 처리
    blurred_nose_region = cv2.GaussianBlur(img, (0, 0), sigmaX=30, sigmaY=30, borderType=cv2.BORDER_DEFAULT)

    # 블러 처리된 영역을 원본 이미지에 적용
    img = np.where(mask[:, :, None] == 255, blurred_nose_region, img)

#     # 결과 이미지 표시
#     cv2.imshow("lsh", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return img

#     output_path = os.path.join(output_folder, "[27, 35, 34, 33, 32, 31].jpg")
#     cv2.imwrite(output_path, img2)

def blur_eyes(img, landmarks):
    # 이미지를 PIL에서 NumPy 배열로 변환하고 데이터 유형을 int로 변환
    img = np.array(img, dtype=np.uint8)

    # 블러 처리 함수
    def apply_blur(image, region, kernel_size=(25, 25)):
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
        image[y:y+h, x:x+w] = blurred_roi

    # 경계 상자 확장 함수
    def expand_box(image, box, factor_x, factor_y):
        x, y, w, h = box
        x = max(x - int(w * factor_x), 0)
        y = max(y - int(h * factor_y), 0)
        w = min(int(w * (1 + 2 * factor_x)), image.shape[1] - x)
        h = min(int(h * (1 + 2 * factor_y)), image.shape[0] - y)
        return (x, y, w, h)

    # 오른쪽 눈
    right_eye_indices = list(range(36, 42))
    # 왼쪽 눈
    left_eye_indices = list(range(42, 48))

    # 오른쪽 눈의 경계 상자 계산
    right_eye_x = [landmarks[i][0] for i in right_eye_indices]
    right_eye_y = [landmarks[i][1] for i in right_eye_indices]
    right_eye_x_min = min(right_eye_x)
    right_eye_x_max = max(right_eye_x)
    right_eye_y_min = min(right_eye_y)
    right_eye_y_max = max(right_eye_y)

    # 왼쪽 눈의 경계 상자 계산
    left_eye_x = [landmarks[i][0] for i in left_eye_indices]
    left_eye_y = [landmarks[i][1] for i in left_eye_indices]
    left_eye_x_min = min(left_eye_x)
    left_eye_x_max = max(left_eye_x)
    left_eye_y_min = min(left_eye_y)
    left_eye_y_max = max(left_eye_y)

    # 경계 상자 확장 (위아래로 확장)
    expansion_factor_x = 0.3  # 가로 확장 비율을 조절하여 원하는 크기로 조정
    expansion_factor_y = 0.6  # 세로 확장 비율을 조절하여 원하는 크기로 조정
    
    # 오른쪽 눈과 왼쪽 눈의 경계 상자를 계산하여 할당
    right_eye_box = expand_box(img, (right_eye_x_min, right_eye_y_min, right_eye_x_max - right_eye_x_min, right_eye_y_max - right_eye_y_min), expansion_factor_x, expansion_factor_y)
    left_eye_box = expand_box(img, (left_eye_x_min, left_eye_y_min, left_eye_x_max - left_eye_x_min, left_eye_y_max - left_eye_y_min), expansion_factor_x, expansion_factor_y)

    # 경계 상자 좌표를 정수로 변환
    right_eye_box = tuple(map(int, right_eye_box))
    left_eye_box = tuple(map(int, left_eye_box))

    # 블러 처리 강화
    apply_blur(img, right_eye_box, kernel_size=(85, 85))
    apply_blur(img, left_eye_box, kernel_size=(85, 85))

    # 결과 이미지 반환
    return img

def total_blur(img, landmark):

    img = blur_nose(img, landmark)
    img = blur_mouse(img, landmark) 
    img = blur_eyes(img,landmark)

    return img
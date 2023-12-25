# FingerBeam
**KISIA AI 보안 기술개발 인력양성 최우수상 수상**

### mAIster 팀 : 리뷰 내 개인정보 마스킹 기술 개발

리뷰 이미지 개인정보 마스킹 모델 개발 및 리뷰 서버 구현
- 안전하고 편안한 환경 제공
- 데이터 무단 이용 방지
- 사진 보정의 자동화
- 작성 시간 단축


-------------------

**메인 기능**

- **리뷰 서버 구현 및 데이터베이스 관리** : 파이썬 Flask 이용한 서버 구축, MongoDB 이용한 데이터 베이스 구축

- **리뷰사진 내 배경 블러 모델** : MRCNN pre-trained 모델 사용

- **리뷰사진 내 얼굴 블러 모델** : detector-MTCNN, landmark-MobileNet pre-trained 모델 사용



**4. 실행 화면**

<br>

- 리뷰 서버 화면

구축한 리뷰 서버

![1](https://github.com/zn122/DetectPrivacyInReviews/blob/master/img/review_server.jpg)

<br>

- 리뷰 작성형태

제목, 별점, 이미지, 배송, 리뷰 에 대한 평가 작성 기능

![2](https://github.com/zn122/DetectPrivacyInReviews/blob/master/img/wirte_review.jpg)

<br>

- 블러 처리된 리뷰 이미지

이미지 업로드 시 자동으로 블러처리 진행하여 서버에 저장

![3](https://github.com/zn122/DetectPrivacyInReviews/blob/master/img/blur_img.jpg)

<br>

- 저장된 리뷰

배경, 얼굴 블러 처리 후 저장됨, 사람 인식이 되지 않을 경우 제품 이미지 그대로 저장

![4](https://github.com/zn122/DetectPrivacyInReviews/blob/master/img/check_review.jpg)



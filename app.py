from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os
import sys
from datetime import datetime
from flask import send_from_directory
from keras import backend as K
import time

from model.mrcnn import utils
from model.coco import coco2
import model.mrcnn.model as modellib
from face_model.models.basenet import MobileNet_GDConv
import torch


from model.mask_rcnn import BackgroundMasking
from face_model.face_detect_blur import FaceMasking

app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client.test_review

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
NUM = 0

#모델 그냥 서버 구축할 때 불러옴.
ROOT_DIR = os.path.abspath("./") #현재 경로
sys.path.append(ROOT_DIR)

MODEL_DIR = os.path.join(ROOT_DIR, "model", "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model", "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# 모델 불러오기 <back>
class InferenceConfig(coco2.CocoConfig): # 'coco.py'의 'CocoConfig' 클래스
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()
back_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config) # Create model object in inference mode.
back_model.load_weights(COCO_MODEL_PATH, by_name=True) # Load weights trained on MS-COCO
back_model.keras_model._make_predict_function()

# 모델불러오기 <face>
if torch.cuda.is_available():
    FaceMasking.map_location = lambda storage, loc: storage.cuda()
else:
    FaceMasking.map_location = 'cpu'
    
face_model = MobileNet_GDConv(136)
face_model = torch.nn.DataParallel(face_model)
script_directory = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_directory, "face_model", "checkpoint","mobilenet_224_model_best_gdconv_external.pth.tar")

checkpoint = torch.load(checkpoint_path, map_location=FaceMasking.map_location)
print(FaceMasking.map_location)
print('Use MobileNet as backbone')
face_model.load_state_dict(checkpoint['state_dict'])

# 클래스 인스턴스 생성
my_instance = BackgroundMasking()
my_instance2 = FaceMasking()
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def make_output(img_list):
    global NUM, back_model, face_model
    print(img_list)
    start = time.time()
    # 클래스 내의 함수 호출
    out_list, NUM = my_instance.background_masking(back_model,img_list, NUM)
    print(out_list)
    # 클래스 내의 함수 호출
    my_instance2.face_masking(face_model,out_list)
    end = time.time()
    print('Time: {:.6f}s.'.format(end - start))
    
    return out_list

@app.route('/')
def home():
    return render_template('index.html')

# 이미지 업로드 API 수정 (여러 이미지 업로드 지원)
@app.route('/upload_image', methods=['POST'])
def upload_image():
    images = request.files.getlist('image')  # 이미지 파일 리스트를 가져옵니다.
    image_filenames = []

    for image in images:
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_filenames.append(filename)

    return jsonify({'msg': '이미지 업로드 완료', 'image_filenames': image_filenames})

@app.route('/submit_review', methods=['POST'])
def write_review():
    title_receive = request.form['title_give']
    delivery_receive = request.form['delivery_give']
    review_receive = request.form['review_give']
    rating_receive = request.form['rating_give']
    current_datetime = datetime.now()

    # 이미지 파일을 업로드하고, 파일 경로를 DB에 저장
    origin_image_paths = []
    for image in request.files.getlist('image'):
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            origin_image_paths.append(image_path.replace("\\", "/"))  # 역슬래시를 슬래시로 변경하여 저장

    image_paths = make_output(origin_image_paths)

    doc = {
        'title': title_receive,
        'delivery': delivery_receive,
        'review': review_receive,
        'images': image_paths,  # 이미지 파일 경로 리스트 저장
        'rating': rating_receive,
        'timestamp': current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    }
    db.review.insert_one(doc)

    reviews = list(db.review.find({}, {'_id': False}))
    return jsonify({'msg': '저장 완료', 'all_reviews': reviews})

# UPLOAD_FOLDER 경로 출력
print(os.path.abspath(app.config['UPLOAD_FOLDER']))

# 경로 존재 확인
upload_folder = app.config['UPLOAD_FOLDER']
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# UPLOAD_FOLDER 경로 출력
print(os.path.abspath(app.config['UPLOAD_FOLDER']))

# 경로 존재 확인
upload_folder = app.config['UPLOAD_FOLDER']
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)


# 리뷰를 표시할 때 이미지 파일을 표시하는 부분
def showReview():
    reviews = list(db.review.find({}, {'_id': False}))

    review_html = ""
    for review in reviews:
        title = review.get('title', '')  # 'title' 정보가 없으면 빈 문자열로 설정
        delivery = review.get('delivery', '')
        review_text = review.get('review', '')
        rating = review.get('rating', '')
        timestamp = review.get('timestamp', '')

        image_tags = ""
        for image_path in review['images']:
            # 이미지 파일 경로를 이용하여 이미지를 표시
            image_tags += f'<img src="/uploads/{os.path.basename(image_path)}" alt="리뷰 이미지" class="review-image">'

        # 리뷰를 HTML 문자열에 추가
        temp_html = f"""
          <tr>
            <td>{title}</td>
            <td>{delivery}</td>
            <td>{review_text}</td>
            <td>{rating}</td>
            <td>{timestamp}</td>
            <td>{image_tags}</td>
            <td><button class="btn btn-danger" onclick="deleteReview('{timestamp}')">삭제</button></td>
          </tr>
        """
        review_html += temp_html

    # 리뷰를 표시하는 영역에 HTML 문자열 삽입
    return f'$("#reviews-box").html(`{review_html}`);'

#리뷰 삭제
@app.route('/delete_review', methods=['POST'])
def delete_review():
    timestamp = request.form['timestamp']

    # 해당 timestamp를 가진 리뷰를 삭제합니다.
    db.review.delete_one({'timestamp': timestamp})

    return jsonify({'msg': '리뷰 삭제 완료'}), 200


@app.route('/get_reviews', methods=['GET'])
def read_reviews():
    reviews = list(db.review.find({}, {'_id': False}))
    return jsonify({'all_reviews': reviews})

# 이미지 파일 제공을 위한 엔드포인트 추가
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
        print(os.path.abspath(app.config['UPLOAD_FOLDER']))

        # 경로 존재 확인
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        app.run('0.0.0.0', port=5000)
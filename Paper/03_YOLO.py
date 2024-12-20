# 이미지 생성이 처음 한두 번은 됐다가 안 됨.

# pip install ultralytics
# pip install cmake
# cmake --version
# pip install dlib
# pip install face_recognition

# print(torch.__version__)         # 2.5.0+cu124
# print(torchvision.__version__)   # 0.20.0+cu124

import cv2
from ultralytics import YOLO
import face_recognition
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# YOLOv11 모델 불러오기
model = YOLO("yolo11x.pt").to("cuda")  # GPU 사용

# 얼굴 데이터베이스 (이름과 인코딩된 얼굴)
known_faces = {
    "Jin": face_recognition.face_encodings(face_recognition.load_image_file("C:/Users/AI5/Documents/카카오톡 받은 파일/hj.jpg"))[0],
}

# 이미지 파일 경로
image_path = "C:/Users/AI5/Documents/카카오톡 받은 파일/hj.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("이미지를 불러오는 데 실패했습니다. 경로를 확인하세요.")
    exit()

# 이미지 사이즈 조정
frame = cv2.resize(frame, (800, 800))  # 크기를 600x900으로 조정

# YOLO 모델로 사람(Person)만 탐지
results = model.predict(frame)
detections = results[0].boxes.xyxy  # 바운딩 박스 좌표 가져오기
classes = results[0].boxes.cls  # 각 바운딩 박스에 대한 클래스 정보 가져오기

# 사람 클래스(일반적으로 YOLO에서 사람 클래스는 0)만 필터링
person_class_id = 0  # YOLO 클래스 ID에서 '사람'은 0번 클래스
person_detections = [box for box, cls in zip(detections, classes) if cls == person_class_id]

# 얼굴 인식 및 이름 표시
for box in person_detections:
    x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표
    cropped_face = frame[y1:y2, x1:x2]

    # 얼굴 인식 수행
    face_encoding = face_recognition.face_encodings(cropped_face)
    if face_encoding:
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding[0])
        if True in matches:
            name = list(known_faces.keys())[matches.index(True)]
        else:
            name = "Unknown"
    else:
        name = "안혜지"  # 얼굴이 인식되지 않을 경우 기본 이름 설정

    # 바운딩 박스 그리기 (OpenCV)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색 바운딩 박스

# OpenCV 이미지를 PIL 이미지로 변환
pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_img)

# 폰트 경로 지정 (Windows 시스템의 맑은 고딕 폰트 사용)
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 20)  # 폰트 크기 20으로 설정

# 텍스트 추가 (바운딩 박스 위에 텍스트 추가)
for box in person_detections:
    x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표
    draw.text((x1, y1 - 25), name, font=font, fill=(255, 255, 255))  # 흰색 텍스트

# PIL 이미지를 다시 OpenCV 이미지로 변환
frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 결과 이미지 창에 표시
cv2.imshow("YOLO + Face Recognition", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

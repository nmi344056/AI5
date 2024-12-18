import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np

# COCO 데이터셋 가중치를 사용하여 Mask R-CNN 모델 불러오기
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 모델을 평가 모드로 전환

# 장치 설정 (GPU를 사용할 수 있으면 GPU로, 그렇지 않으면 CPU 사용)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)
path = 'C:/ai5/_data/rcnn/aa/'
# 분석할 이미지 파일 경로 지정
test_image_path = path + "22.jpg"

# 이미지 로드 및 색상 채널 변환
image = cv2.imread(test_image_path)
image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB 형식으로 변환
image_color = image_color.astype(np.float32) / 255.0  # 픽셀 값을 [0, 1] 범위로 스케일링
image_tensor = torch.tensor(image_color).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # Tensor로 변환

# 모델을 통해 예측 수행
with torch.no_grad():
    pred = model(image_tensor)

# 예측 결과 가져오기
pred_boxes = pred[0]["boxes"].cpu().numpy()  # 바운딩 박스 좌표
pred_scores = pred[0]["scores"].cpu().numpy()  # 신뢰도 점수
pred_masks = pred[0]["masks"].cpu().numpy()  # 마스크 데이터

# 표시 기준으로 사용할 신뢰도 임계값
threshold = 0.5
sample = (image_color * 255).astype(np.uint8)  # 이미지 데이터를 [0, 255] 범위로 변환
for i, box in enumerate(pred_boxes):
    if pred_scores[i] >= threshold:
        # 객체를 감싸는 사각형 그리기
        box = box.astype(int)
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        
        # 무작위 색상 값 생성
        random_color = np.random.randint(0, 257, size=3).tolist()  # RGB 형태의 임의 색상 지정
        
        # 객체 마스크 생성
        mask = pred_masks[i, 0] > threshold
        colored_mask = np.zeros_like(sample, dtype=np.uint8)
        colored_mask[mask] = random_color  # 마스크에 무작위 색상 적용
        sample = cv2.addWeighted(sample, 1, colored_mask, 0.5, 0)  # 원본 이미지와 마스크 합성

# 최종 결과 이미지 출력
plt.imshow(sample)
plt.axis("off")
plt.show()

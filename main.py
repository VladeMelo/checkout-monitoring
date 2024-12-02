import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

class_names = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
  'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
  'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class_names_goal = ['person']

model = YOLO('yolov8m.pt')

video = cv2.VideoCapture('checkout.mp4')

width = 1280
height = 720

top_left_checkout_area, bottom_right_checkout_area = (420, 200), (700, 400)

fps = int(video.get(cv2.CAP_PROP_FPS))
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('checkout_monitoring.mp4', fourcc, fps, (width, height))

processing_time = 0

while True:
  success, frame = video.read()

  if not success:
    break

  frame = cv2.resize(frame, (width, height))

  results = model(frame, stream=True)

  detections = []

  for result in results:
    for box in result.boxes:
      class_name = class_names[int(box.cls[0])]

      if not class_name in class_names_goal:
        continue

      confidence = round(float(box.conf[0]) * 100, 2)

      if confidence < 30:
        continue

      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      if x2 > top_left_checkout_area[0]:
        if x1 < bottom_right_checkout_area[0]:
          text = 'Processing'
          color = (140, 57, 31)
          color_text = (255, 255, 255)

          processing_time += 1
        else:
          text = 'Cashier'
          color = (255, 255, 255)
          color_text = (0, 0, 0)
      else:
        text = 'Waiting'
        color = (0, 0, 255)
        color_text = (255, 255, 255)

      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
      cvzone.putTextRect(frame, text, (x1, y1), 1, 1, offset=0, border=2, colorR=color, colorB=color, colorT=color_text, font=cv2.FONT_HERSHEY_DUPLEX)

  total_seconds = processing_time / fps
  minutes, seconds = divmod(int(total_seconds), 60)

  processing_time_text = f'Processing Time: {minutes:02}:{seconds:02}'

  cv2.rectangle(frame, top_left_checkout_area, bottom_right_checkout_area, (255, 255, 255), 2)
  cvzone.putTextRect(frame, processing_time_text, ((bottom_right_checkout_area[0] + top_left_checkout_area[0]) // 2 - 90, (bottom_right_checkout_area[1] + top_left_checkout_area[1]) // 2 + 40), 1, 1, offset=4, colorR=(255, 255, 255), colorT=(0, 0, 0))

  output_video.write(frame)

  cv2.imshow('Image', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

output_video.release()
video.release()

cv2.destroyAllWindows()
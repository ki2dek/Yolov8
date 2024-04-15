import cv2
from ultralytics import YOLO

# model = YOLO('yolov8s.pt')

model = YOLO("yolov8n.pt")
file_path = "000000002076.jpg"
results = model.predict(rf'datasets\coco\test2017\{file_path}')




plots = results[0].plot()

cv2.imshow('show',plots)

cv2.waitKey(0)  # 무한 대기, 사용자가 키를 누를 때까지

cv2.destroyAllWindows()



import cv2
from ultralytics import YOLO

# model = YOLO('yolov8s.pt')

model = YOLO(rf"runs\detect\train\weights\best.pt")
file_path = "72_png_jpg.rf.88b1eabba0f0595962ba4ac68c89394f.jpg"
results = model(rf'datasets\IU\test\images\{file_path}') # conf=0.2, iou ..

plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()

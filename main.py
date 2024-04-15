import cv2
import ultralytics
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
file_path = "7244811-hd_1080_1920_25fps.mp4"
results = model.predict(rf'{file_path}')


# print(type(model.names), len(model.names))
# print('model.names',model.names)
plots = results[0].plot()

cv2.imshow('show',plots)

cv2.waitKey(0)  # 무한 대기, 사용자가 키를 누를 때까지

cv2.destroyAllWindows()



    
    
    
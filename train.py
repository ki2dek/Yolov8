import cv2
import ultralytics
from ultralytics import YOLO
import multiprocessing

def trainIU():
    model = YOLO("yolov8n.pt")
    data_path = "data.yaml"
    model.train(data=data_path, epochs=150, patience=30, batch=32, imgsz=416)
    # model.train(data=data_path, epochs=100, patience=30, batch=32, imgsz=416, device='cuda:0')


    print(type(model.names), len(model.names))
    print('model.names',model.names)


def trainCOCO():
    model = YOLO("yolov8n.pt")
    data_path = "coco_data.yaml"
    model.train(data=data_path, epochs=150, patience=30, batch=32, imgsz=416)
    # model.train(data=data_path, epochs=100, patience=30, batch=32, imgsz=416, device='cuda:0')


    print(type(model.names), len(model.names))
    print('model.names',model.names)




def worker():
    print("Worker Function")

if __name__ == '__main__':
    # 윈도우에서 multiprocessing을 안전하게 사용하기 위한 필수 조치
    multiprocessing.freeze_support()
    trainIU()
    # trainCOCO()
    # p = multiprocessing.Process(target=worker)
    # p.start()
    # p.join()
    
    
    
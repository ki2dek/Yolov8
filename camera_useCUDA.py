import cv2
from ultralytics import YOLO
import torch

def detect_video():
    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')
    
    # CUDA 사용 가능 여부 확인 및 모델을 GPU로 옮기기
    print('torch.cuda.is_available()',torch.cuda.is_available())
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 카메라 인덱스 확인
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    print("Available camera indexes:", arr)
    
    cap = cv2.VideoCapture(1)  # USB 카메라
    cap.set(cv2.CAP_PROP_FPS, 30)  # 카메라 FPS 설정

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    tick_count = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 조정
        frame = cv2.resize(frame, (1000, 800))

        # GPU로 데이터 전송
        if torch.cuda.is_available():
            frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

        # 프레임에 대한 예측 수행
        results = model.predict(frame)
        plots = results[0].plot()
        
        # FPS 계산
        ticks_now = cv2.getTickCount()
        time_spent = (ticks_now - tick_count) / cv2.getTickFrequency()
        fps = 1 / time_spent
        tick_count = ticks_now

        # 화면에 FPS 표시
        cv2.putText(plots, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 화면에 결과 표시
        plots = plots.cpu().numpy() if torch.cuda.is_available() else plots
        cv2.imshow('Detection', plots)

        if cv2.waitKey(1) == ord('q'):  # 'q'를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

detect_video()

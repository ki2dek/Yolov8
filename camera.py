import cv2
from ultralytics import YOLO

def detect_video():
    model = YOLO('yolov8n.pt')  # yolo 모델 로드
    # model = YOLO(rf"runs\detect\train\weights\best.pt") # 아이유 모델 로드
    
    
    
    # 사용 가능한 카메라 목록을 확인하기
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

    print("Available camera indexes:", arr)  # 사용 가능한 카메라 인덱스 출력
    
    
    cap = cv2.VideoCapture(1)  # 0: 기본 카메라 , 1: usb 카메라

    
    # cap.set(cv2.CAP_PROP_FPS, 60)   # 카메라가 60 FPS를 지원하는지 확인 후 설정
    cap.set(cv2.CAP_PROP_FPS, 30)   # 카메라가 30 FPS를 지원하는지 확인 후 설정

    frame_time = int((1/30) * 1000)  # 30 FPS에 해당하는 프레임 시간 계산

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tick_count = cv2.getTickCount()
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 조정 (선택적)
        frame = cv2.resize(frame, (1000, 800))

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

        cv2.imshow('Detection', plots)
        if cv2.waitKey(frame_time) == ord('q'):  # 'q'를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

detect_video()

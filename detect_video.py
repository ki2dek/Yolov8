import cv2
from ultralytics import YOLO

def detect_video(video_path):
    # model = YOLO('yolov8n.pt')  # yolo 모델 로드
    model = YOLO(rf"runs\detect\train\weights\best.pt") # 아이유 모델 로드
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    tick_count = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 조절할 프레임의 너비와 높이를 설정합니다.
        # desired_width = 720
        # desired_height = 1000
        desired_width = 1000
        desired_height = 800
        
        frame = cv2.resize(frame, (desired_width, desired_height))

        # 프레임에 대한 예측 수행
        results = model.predict(frame)

        # # 탐지된 객체의 바운딩 박스 및 클래스 정보 추출
        # detections = results.pred[0]  # 결과에서 탐지된 객체 정보 가져오기

        # # 결과 추출 및 바운딩 박스 그리기
        # for *xyxy, conf, cls in detections:
        #     x1, y1, x2, y2 = map(int, xyxy)
        #     label = f"{model.names[int(cls)]} {conf:.2f}"
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # cv2.imshow('Video', frame)
        
        plots = results[0].plot()
        
        # FPS 계산
        ticks_now = cv2.getTickCount()
        time_spent = (ticks_now - tick_count) / cv2.getTickFrequency()
        fps = 1 / time_spent
        tick_count = ticks_now

        # 화면에 FPS 표시
        cv2.putText(plots, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('show',plots)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'): # 40프레임
        # if cv2.waitKey(33) == ord('q') or cv2.waitKey(33) == ord('Q'): # 30프레임
        # if cv2.waitKey(100) == ord('q') or cv2.waitKey(100) == ord('Q'): # 10프레임
            break

    cap.release()
    cv2.destroyAllWindows()



# 비디오 파일 경로 지정
# video_path = 'KakaoTalk_20240324_000701696.mp4'
# video_path = '2836305-uhd_3840_2160_24fps.mp4'
video_path = '7244811-hd_1080_1920_25fps.mp4'
# video_path = '6562756-hd_720_1284_30fps.mp4'
detect_video(video_path)

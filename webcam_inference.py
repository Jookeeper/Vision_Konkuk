# webcam_infefence.py
import cv2
from ultralytics import YOLO
from DB import RISK_DATABASE

def main():
    # 1. 학습 중인 커스텀 가중치 파일 로드하기
    model = YOLO('./runs/detect/train/weights/best.pt')

    # 2. 노트북 기본 웹캠 열기 (나중에 외장 캠으로 변경)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    print("웹캠이 성공적으로 열렸습니다. 물체 인식 중...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 3. 웹캠 프레임에서 물체 인식 수행
        results = model(frame, stream=True)

        # 4. 인식된 물체마다 DB 매핑 및 위험도 정보 출력
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 클래스 정보 추출
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = box.conf[0].item()

                print(f"\n[AI 인식 성공] 물체: {class_name} (확률: {confidence:.2f})")

                # 사물의 실시간 바운딩 박스 좌표 확보 및 중심점 계산
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int(x1 + x2) // 2
                center_y = int(y1 + y2) // 2

                # ai 인식 결과와 DB 대조 -> 의사결정 시각화
                status_text = "DB 미등록 물체"
                robot_cmd = "HOLD"
                color = (0, 255, 0)  # DB 미등록, 기본 녹색 (안전)

                if class_name in RISK_DATABASE:
                    db_info = RISK_DATABASE[class_name]
                    status_text = db_info['status']
                    robot_cmd = db_info['robot_action_cmd']

                    # 위험도에 따른 화면 표시 색상 변화
                    if db_info['level'] == 3:
                        color = (0, 0, 255)  # 위험(빨강)
                    elif db_info['level'] == 2:
                        color = (0, 255, 255)  # 주의(노랑)
                    elif db_info['level'] == 1:
                        color = (255, 255, 0)  # 관심(하늘색)
                
                # 화면에 바운딩 박스, 중심점, 로봇 제어 명령 텍스트 그리기
                
                # 사각형 박스 그리기 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 중심점 표시 
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                # 텍스트 정보 표시 (물체 이름, 위험도 상태, 로봇
                info_str = f"{class_name} [{status_text}] CMD : {robot_cmd}"
                coord_str = f"Center : ({center_x}, {center_y})"

                cv2.putText(frame,info_str,(x1,y1-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                cv2.putText(frame,coord_str,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,2)
            
            # 모니터링 창에 실시간 화면 출력
            cv2.imshow('Webcam Object Detection (Test)', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료 키 입력 감지. 프로그램을 종료합니다.")
                break
    # 웹캠과 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


                                

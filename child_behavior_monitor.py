# child_behavior_monitor.py
import cv2
import mediapipe as mp
from ultralytics import YOLO
from DB import RISK_DATABASE  # 우리가 고쳐둔 진짜 위험도 DB 파일

# 두 바운딩 박스가 서로 겹치는지(충돌하는지) 검사하는 함수
def is_overlapping(boxA, boxB):
    # box = (x1, y1, x2, y2)
    return not (boxA[2] < boxB[0] or boxA[0] > boxB[2] or boxA[3] < boxB[1] or boxA[1] > boxB[3])

def main():
    # 1. 현재 학습 중인 '우리만의 진짜 AI 뇌(가중치)' 로드
    # [주의] 30에폭 학습이 완전히 끝나야 이 best.pt 파일이 생성되어 에러 없이 실행됩니다!
    model = YOLO('./runs/detect/train/weights/best.pt')

    # 2. MediaPipe Holistic 가동 (얼굴 랜드마크 + 양손 추적)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    print("👶 [행동 감시] 커스텀 위험물(Car, Panda, Rabbit 등) 3대 박스 충돌 검증 가동...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # 거울 모드
        h, w, _ = frame.shape
        
        # 실시간 상태 변수 초기화
        custom_object_boxes = []  # 📦 진짜 위험물 박스 저장용
        mouth_box = None          # 👄 입 박스 저장용
        hand_boxes = []           # ✋ 손 박스 저장용
        
        holding_object_info = None # 🎯 현재 손에 쥔 물체의 DB 정보 저장용
        is_holding = False         # 물체 파지 여부 플래그

        # ----------------------------------------------------------------
        # 📦 [박스 1] YOLOv8 커스텀 모델로 '진짜 우리 위험물'만 추출
        # ----------------------------------------------------------------
        yolo_results = model(frame, stream=True, verbose=False)
        for result in yolo_results:
            for box in result.boxes:
                c_id = int(box.cls[0])
                class_name = model.names[c_id]
                
                # 핸드폰 등 잡다한 사물 무시 ➡️ 오직 우리 RISK_DATABASE에 등록된 5가지 물체만 필터링!
                if class_name in RISK_DATABASE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    custom_object_boxes.append((x1, y1, x2, y2, class_name))
                    
                    # 바닥 탐지 상태 색상 정의 (기본 녹색)
                    db_info = RISK_DATABASE[class_name]
                    color = (255, 255, 0) # 관심(하늘색)
                    if db_info['level'] == 3: color = (0, 0, 255) # 위험(빨강)
                    elif db_info['level'] == 2: color = (0, 255, 255) # 주의(노랑)

                    # 사물 박스 시각화
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} (Floor)", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ----------------------------------------------------------------
        # 👄 [박스 2] MediaPipe로 '입(Mouth)' 바운딩 박스 생성
        # ----------------------------------------------------------------
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = holistic.process(image_rgb)

        if mp_results.face_landmarks:
            ml = mp_results.face_landmarks.landmark[13] # 윗입술 중앙점
            mx, my = int(ml.x * w), int(ml.y * h)
            # 입을 감싸는 보이지 않는 가상의 센서 박스 가동 (가로 60, 세로 40)
            mouth_box = (mx - 30, my - 20, mx + 30, my + 20)
            cv2.rectangle(frame, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]), (255, 0, 255), 1)

        # ----------------------------------------------------------------
        # ✋ [박스 3] MediaPipe로 '손(Hand)' 영역 바운딩 박스 생성
        # ----------------------------------------------------------------
        for hand_landmarks in [mp_results.right_hand_landmarks, mp_results.left_hand_landmarks]:
            if hand_landmarks:
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                hx1, hx2 = int(min(x_list) * w), int(max(x_list) * w)
                hy1, hy2 = int(min(y_list) * h), int(max(y_list) * h)
                hand_boxes.append((hx1, hy1, hx2, hy2))
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 255), 1) # 흰색 손 박스

        # ----------------------------------------------------------------
        # 🚨 [의사결정 및 연동] 3대 박스 충돌 판단 알고리즘
        # ----------------------------------------------------------------
        # 1단계: 내 손 박스가 우리가 가르친 '커스텀 위험물 박스'와 겹쳤는가? (물체 파지 판단)
        for hand in hand_boxes:
            for obj in custom_object_boxes:
                obj_rect = obj[:4] # 좌표만 분리
                obj_name = obj[4]  # 물체 이름 ('Car', 'Rabbit' 등)
                
                if is_overlapping(hand, obj_rect):
                    is_holding = True
                    holding_object_info = RISK_DATABASE[obj_name] # 손에 쥔 물체의 DB 상세 정보 확보
                    holding_object_info['current_name'] = obj_name
                    
                    # 물건을 쥔 손은 주황색 박스로 강조 변경
                    cv2.rectangle(frame, (hand[0], hand[1]), (hand[2], hand[3]), (0, 165, 255), 2)
                    cv2.putText(frame, f"HOLDING: {obj_name}", (hand[0], hand[1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # 2단계: 물체를 확실히 쥔 상태에서, 그 손이 입 박스와 충돌했는가? (최종 삼킴 비상)
        if is_holding and mouth_box and holding_object_info:
            for hand in hand_boxes:
                if is_overlapping(hand, mouth_box):
                    # DB에 저장된 물체별 mouth_action_cmd 및 경고 메시지 꺼내오기
                    mouth_cmd = holding_object_info.get('mouth_action_cmd', 'ALARM')
                    msg = holding_object_info['msg']
                    obj_name = holding_object_info['current_name']
                    
                    # 위험 등급별 UI 알림 차등화 (사용자가 수정한 DB 규칙 반영)
                    if mouth_cmd == 'EMERGENCY_STOP':
                        alert_title = f"🚨🚨 EMERGENCY STOP: {obj_name} TO MOUTH!!! 🚨🚨"
                        alert_color = (0, 0, 255) # 3단계 빨간색 비상바
                    else:
                        alert_title = f"⚠️ WARNING: {obj_name} TO MOUTH DETECTED ⚠️"
                        alert_color = (0, 165, 255) # 1~2단계 주황색 경고바

                    # 화면 상단 경고 팝업 가동
                    cv2.rectangle(frame, (0, 0), (w, 65), alert_color, -1)
                    cv2.putText(frame, alert_title, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(frame, f"DB ACTION: {msg}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
        # 아무것도 안 쥐고 그냥 맨손만 입에 가져갔을 때는 철저히 평온 유지 (알람 무시)
        elif not is_holding and mouth_box:
            for hand in hand_boxes:
                if is_overlapping(hand, mouth_box):
                    cv2.putText(frame, "Bare hand to mouth (Safe/Normal)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Kid Behavior Monitor (Custom Target Lab)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    holistic.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
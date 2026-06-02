# test_inference.py
from ultralytics import YOLO
from DB import RISK_DATABASE

def main():
    # 1. 방금 학습이 완료된 따끈따끈한 custom 모델 불러오기
    model = YOLO('./runs/detect/train/weights/best.pt')

    # 2. 데이터셋의 train/images 폴더 안에 있는 사진 중 하나를 골라 테스트합니다.
    test_image_path = "./train/images/Rabbit_67_jpg.rf.f0cf2bfa3beaacfd90e2875cd3e84474.jpg" 
# C:\Users\porip\OneDrive\Desktop\Capstone_Vision\train\images\
    print(f"🔍 {test_image_path} 이미지 분석 중...")
    results = model(test_image_path)

    # 3. AI가 인식한 결과와 DB 매핑하기
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # AI가 맞춘 클래스 번호를 텍스트 이름으로 변환 (예: 'Car', 'Panda' 등)
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = box.conf[0].item() # 신뢰도 (확률)

            print(f"\n[AI 인식 성공] 물체: {class_name} (확률: {confidence:.2f})")

            # 4. 수정된 RISK_DATABASE에서 해당 물체의 위험도 정보 조회
            if class_name in RISK_DATABASE:
                db_info = RISK_DATABASE[class_name]
                print(f"🚨 [위험도 DB 연동 결과]")
                print(f"   - 위험 레벨: {db_info['level']}")
                print(f"   - 현재 상태: {db_info['status']}")
                print(f"   - 🤖 주행팀 명령 (바닥 발견 시): {db_info['robot_action_cmd']}")
                
                # [업데이트] 입 접촉 시나리오 플래그 출력 (딕셔너리에 없을 경우를 대비해 안전하게 .get() 사용)
                mouth_cmd = db_info.get('mouth_action_cmd', 'ALARM')
                print(f"   - 👶 제어팀 명령 (입 접촉 시): {mouth_cmd}")
                print(f"   - 안내 메시지: {db_info['msg']}")
            else:
                print("⚠️ DB에 등록되지 않은 물체입니다. (기본 모니터링 유지)")

if __name__ == '__main__':
    main()
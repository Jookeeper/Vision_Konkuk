from ultralytics import YOLO

def main():
    # 1. 30에폭 학습이 완료된 커스텀 가중치 파일 로드
    model = YOLO('./runs/detect/train/weights/best.pt')
    
    print("=== 모의 데이터셋 기반 AI 모델 성능 검증 시작 ===")
    
    # 2. data.yaml에 정의된 valid 데이터셋을 바탕으로 정량적 성능 검증 수행
    # split='val' 또는 split='test'로 설정하여 검증/테스트 데이터 지정 가능
    metrics = model.val(split='val') # 검증 데이터
    test_metrics = model.val(split='test') # 테스트 데이터 (필요 시)
    
    # 3. 출력
    print("\n==================================================")
    print("       📊 [캡스톤디자인 비전 성능 검증 결과]       ")
    print("==================================================")
    
    # 각 클래스별 통합 지표 추출 (검증된 데이터 기준)
    precision_val = metrics.results_dict.get('metrics/precision(B)', 0)
    recall_val = metrics.results_dict.get('metrics/recall(B)', 0)
    map50_val = metrics.results_dict.get('metrics/mAP50(B)', 0)
    map50_95_val = metrics.results_dict.get('metrics/mAP50-95(B)', 0)
    print(f"val 데이터셋 기준으로 모델 성능 지표:")
    print(f"🔹 Precision (정밀도) : {precision_val:.4f}  (모델이 탐지한 것 중 진짜 맞춘 비율)")
    print(f"🔹 Recall    (재현율) : {recall_val:.4f}  (실제 위험물 중 모델이 찾아낸 비율)")
    print(f"🔹 mAP@50             : {map50_val:.4f}  (정확도 지표, 0.5 임계치 기준)")
    print(f"🔹 mAP@50-95          : {map50_95_val:.4f}  (엄격한 다중 임계치 종합 점수)")
    print("==================================================")
    print("💡 검증 완료! 상세 그래프는 'runs/detect/val/' 폴더를 확인하세요.")
    print("\n================================================")

    # 각 클래스별 통합 지표 추출 (테스트된 데이터 기준, 필요시에 사용)
    precision_test = test_metrics.results_dict.get('metrics/precision(B)', 0)
    recall_test = test_metrics.results_dict.get('metrics/recall(B)', 0)
    map50_test = test_metrics.results_dict.get('metrics/mAP50(B)', 0)
    map50_95_test = test_metrics.results_dict.get('metrics/mAP50-95(B)', 0)

    print(f"test 데이터셋 기준으로 모델 성능 지표:")
    print(f"🔹 Precision (정밀도) : {precision_test:.4f}  (모델이 탐지한 것 중 진짜 맞춘 비율)")
    print(f"🔹 Recall    (재현율) : {recall_test:.4f}  (실제 위험물 중 모델이 찾아낸 비율)")
    print(f"🔹 mAP@50             : {map50_test:.4f}  (정확도 지표, 0.5 임계치 기준)")
    print(f"🔹 mAP@50-95          : {map50_95_test:.4f}  (엄격한 다중 임계치 종합 점수)")
    print("==================================================")
    print("💡 검증 완료! 상세 그래프는 'runs/detect/test/' 폴더를 확인하세요.")


if __name__ == '__main__':
    main()
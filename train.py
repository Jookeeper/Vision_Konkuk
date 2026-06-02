# train.py
from ultralytics import YOLO

def main():
    # 1. 구글이 기본 제공하는 사전 학습된 YOLOv8 초경량 모델(Nano) 불러오기
    # 최초 실행 시 모델 파일이 자동으로 다운로드됩니다.
    model = YOLO('yolov8n.pt')

    # 2. 다운로드받은 모의 데이터셋으로 딱 3바퀴(epochs)만 테스트 학습 돌리기
    # epochs=3으로 설정해서 코드가 에러 없이 끝까지 도는지 빠르게 확인합니다.
    print("🚀 YOLOv8 모의 데이터셋 학습 테스트를 시작합니다...")
    model.train(data='./data.yaml', epochs=30, imgsz=640)  
    print("🎉 테스트 학습이 성공적으로 끝났습니다!")

if __name__ == '__main__':
    main()


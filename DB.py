# 모의 데이터셋(원숭이, 인형 등) 맞춤형 위험도 차등 대조 테이블
RISK_DATABASE = {
    # 3단계: 선제거 O + 입에 대면 '즉시 최고 등급 비상 정지' (예: 단추형 건전지 대용)
    'Car': {
        'level': 3,
        'status': '위험(Emergency)',
        'robot_action_cmd': 'REMOVE',         # 발견 즉시 선제 제거
        'mouth_action_cmd': 'EMERGENCY_STOP',  # 입에 대면 즉시 급정지 및 사이렌
        'msg': '단추형 건전지(Car) 감지! 화학 화상 위험이 있으므로 선제거를 개시하며, 입 접촉 즉시 비상 급정지합니다.'
    },
    
    # 2단계-1: 선제거 O + 입에 대면 '일반 경고 알림' (예: 동전 대용)
    'Panda': {
        'level': 2,
        'status': '주의(Caution)',
        'robot_action_cmd': 'REMOVE',         # 발견 즉시 선제 제거
        'mouth_action_cmd': 'ALARM',           # 입에 대면 경고음 발생
        'msg': '동전(Panda) 감지. 기도 폐쇄 위험이 있어 선제거를 개시하며, 입 접촉 시 경고를 울립니다.'
    },
    
    # 2단계-2: 선제거 O + 입에 대면 '일반 경고 알림' (예: 구슬 대용)
    'Rabbit': {
        'level': 2,
        'status': '주의(Caution)',             # 레벨 2에 맞춰 주의(Caution)로 통일했습니다.
        'robot_action_cmd': 'REMOVE',
        'mouth_action_cmd': 'ALARM',
        'msg': '구슬(Rabbit) 감지. 흡인 위험이 있어 선제거를 개시하며, 입 접촉 시 경고를 울립니다.'
    },

    # 2단계-3: 선제거 O + 입에 대면 '일반 경고 알림' (예: 작은 장난감 대용)
    'Teddy': {
        'level': 2,
        'status': '주의(Caution)',
        'robot_action_cmd': 'REMOVE',
        'msg': '작은 장난감(Teddy) 감지. 기도 폐쇄 위험이 있어 선제거를 개시하며, 입 접촉 시 경고를 울립니다.'
    },

    # 1단계: 선제거 X + 대신 평소엔 보기만 하다가 입에 대면 '경고 알림' (예: 일반 큰 장난감 대용)
    'stick': {
        'level': 1,
        'status': '관심(Monitor)',
        'robot_action_cmd': 'MONITOR',        # 평상시에는 치우지 않고 감시만 함
        'mouth_action_cmd': 'ALARM',          # 단, 아이가 입에 가져가면 알림을 울림
        'msg': '일반 장난감(stick) 감지. 치우지 않고 주시하며, 아이가 입에 대는 순간 알림을 가동합니다.'
    }
}
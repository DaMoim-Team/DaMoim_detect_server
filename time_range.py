from datetime import datetime, timedelta
import pytz

def current_range():
    # 한국 시간대 설정
    korea_tz = pytz.timezone('Asia/Seoul')

    # 한국 시간대의 현재 시간에서 5분을 뺀 시간 가져오기(main에서 호출될때, 오차발생 가능성 방지)
    now = datetime.now(korea_tz) - timedelta(minutes=5)
    now_hour = now.hour

    # 다음 시간 계산
    next_hour = (now_hour + 1) % 24

    # 시간 구간 문자열 만들기
    time_range = f"{now_hour:02d}-{next_hour:02d}"
    
    return time_range

import webrtcvad
import collections
import pyaudio
import numpy as np

# ==== PyAudio 설정 ====
FORMAT = pyaudio.paInt16  # 16비트 인코딩
CHANNELS = 1              # 모노 채널
RATE = 16000              # 샘플링 레이트 (16kHz)
FRAME_DURATION = 30       # 프레임 길이 (ms)
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000) # 프레임 크기 (샘플 수)
CHUNK_SIZE = FRAME_SIZE * 1 # 청크 크기

# ===== 에너지 게이트 파라미터 =====
EMA_ALPHA = 0.02           # 소음 바닥값 학습 속도(0.03~0.07 권장)
ENERGY_MARGIN = 10.0       # 바닥 대비 몇 배면 “소리”로 볼지(3.0~5.0 권장)
INITIAL_NOISE_FLOOR = 0.003  # 초기 바닥값(RMS). 너무 크면 시작이 둔감, 너무 작으면 예민
EPS = 1e-12

# ===== 다수결 + 행오버 파라미터 =====
WIN_SIZE = 7        # 최근 프레임 창 크기
WIN_MIN_VOICE = 4   # 시작 트리거: 창 안 '음성' 프레임 최소 개수
HANGOVER_FRAMES = 6 # 종료 조건: 연속 무음 프레임 수

def rms_from_int16(pcm_bytes):
    # int16 리틀엔디언 → float32(-1~1) → RMS
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x) + EPS))


def main():
    vad = webrtcvad.Vad(3)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    print("마이크 녹음을 시작합니다... 'Ctrl+C'를 눌러 종료하세요.")
    noise_floor = INITIAL_NOISE_FLOOR

    # 상태 관리
    votes = collections.deque(maxlen=WIN_SIZE)  # 최근 프레임들의 0/1
    triggered = False
    hang = 0

    try:
        while True:
            # 마이크에서 오디오 데이터 읽기
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # 읽은 데이터가 비어있으면 종료
            if not data:
                break

            e = rms_from_int16(data)
            
            # 2) 노이즈 바닥값 업데이트(느리게, 너무 큰 프레임은 제외)
            #    - “정말 배경 소음으로 보이는 구간”에서만 EMA 반영
            if e < noise_floor * 1.5:
                noise_floor = (1 - EMA_ALPHA) * noise_floor + EMA_ALPHA * e

            # 3) 에너지 게이트: 바닥보다 충분히 크면 후보
            is_loud_enough = e > (noise_floor * ENERGY_MARGIN)

            # 4) (권장) AND 결합: 에너지+VAD 둘 다 만족해야 '음성'
            if is_loud_enough and vad.is_speech(data, RATE):
                print("🚨 음성(에너지+VAD) 감지")
            else:
                print("침묵/잡음  "
                      f"(rms={e:.4f}, floor={noise_floor:.4f}, "
                      f"ratio={e/max(noise_floor, EPS):.2f}x)")

    except KeyboardInterrupt:
        print("\n녹음을 종료합니다.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    main()
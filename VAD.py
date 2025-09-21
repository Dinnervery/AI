import webrtcvad
import collections
import sys
import signal
import pyaudio
import numpy as np

# PyAudio 설정
FORMAT = pyaudio.paInt16  # 16비트 인코딩
CHANNELS = 1              # 모노 채널
RATE = 16000              # 샘플링 레이트 (16kHz)
FRAME_DURATION = 30       # 프레임 길이 (ms)
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000) # 프레임 크기 (샘플 수)
CHUNK_SIZE = FRAME_SIZE * 1 # 청크 크기

def main():
    # VAD 인스턴스 생성 및 모드 설정
    # 모드 0: 공격적인 모드 (가장 엄격하게 음성 감지)
    # 모드 1: 중간
    # 모드 2: 보통
    # 모드 3: 가장 공격적인 모드 (가장 덜 엄격하게 음성 감지)
    vad = webrtcvad.Vad(3)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    print("마이크 녹음을 시작합니다... 'Ctrl+C'를 눌러 종료하세요.")

    try:
        while True:
            # 마이크에서 오디오 데이터 읽기
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # 읽은 데이터가 비어있으면 종료
            if not data:
                break
            
            # 음성 감지
            is_speech = vad.is_speech(data, RATE)

            if is_speech:
                print("🚨 음성이 감지되었습니다!")
            else:
                print("침묵..")

    except KeyboardInterrupt:
        print("\n녹음을 종료합니다.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    main()
import webrtcvad
import collections
import pyaudio
import wave
import tempfile
from pathlib import Path
import numpy as np

# 빠른 STT를 위해 faster_whisper 사용 권장 (pip install faster-whisper)
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    print("[경고] faster-whisper가 설치되어 있지 않습니다. STT는 작동하지 않습니다.")
    WHISPER_AVAILABLE = False


# ==== PyAudio 설정 ====
FORMAT = pyaudio.paInt16  # 16비트 인코딩
CHANNELS = 1              # 모노 채널
RATE = 16000              # 샘플링 레이트 (16kHz)
FRAME_DURATION = 30       # 프레임 길이 (ms)
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 프레임 크기 (샘플 수)
CHUNK_SIZE = FRAME_SIZE * 1  # 청크 크기

# ===== 에너지 게이트 파라미터 =====
EMA_ALPHA = 0.02          # 소음 바닥값 학습 속도
ENERGY_MARGIN = 10.0       # 바닥 대비 몇 배의 소리를 인식할지
INITIAL_NOISE_FLOOR = 0.003  # 초기 바닥값(RMS)
EPS = 1e-12

# ===== 다수결 + 행오버 파라미터 =====
WIN_SIZE = 7         # 최근 프레임 창 크기
WIN_MIN_VOICE = 4    # 시작 트리거: 창 안 '음성' 프레임 최소 개수
HANGOVER_FRAMES = 25  # 종료 조건: 연속 무음 프레임 수

# ===== 프리-롤 버퍼 파라미터 =====
# 발화 시작 시 앞부분이 잘리는 것을 방지하기 위해, 시작 전 5 프레임(150ms)을 저장
PRE_ROLL_FRAMES = 5


def rms_from_int16(pcm_bytes):
    # int16 리틀엔디언 → float32(-1~1) → RMS
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x) + EPS))


def write_wav_int16(path, pcm_bytes, channels=1, rate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)


# ====== Whisper 모델 로드 및 STT 처리 ======
def get_whisper_model():
    """Whisper 모델을 한 번만 로드하여 반환합니다."""
    if not WHISPER_AVAILABLE:
        return None
    
    print("🧠 Whisper 모델 로드 중... (medium, CPU 최적화)")
    #int8 - 호환성, 메모리/속도 유리   float8 - 안전, 속도 느림, 메모리 사용 높음
    return WhisperModel("medium", device="cpu", compute_type="int8")


def transcribe_with_whisper(model, wav_path, language="ko"):
    """로드된 모델을 사용하여 WAV 파일을 STT 처리합니다."""
    if model is None:
        return "[STT 모듈 없음]"
        
    try:
        segments, info = model.transcribe(wav_path, language=language, vad_filter=False)
        text = "".join(seg.text for seg in segments)
        return text.strip()
    except Exception as e:
        return f"[STT 오류: {e}]"


# ========================================================
# 메인 VAD 실행 로직
# ========================================================
def main():
    vad = webrtcvad.Vad(3)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)
    
    whisper_model = get_whisper_model()  # 🌟 Whisper 모델을 프로그램 시작 시 한 번만 로드

    print("마이크 녹음을 시작합니다... 'Ctrl+C'를 눌러 종료하세요.")
    noise_floor = INITIAL_NOISE_FLOOR

    # 상태 관리
    votes = collections.deque(maxlen=WIN_SIZE)  # 최근 프레임들의 0/1
    triggered = False
    hang = 0
    
    # 🌟 프리-롤 버퍼: 발화 시작 전의 오디오 데이터를 저장하여 시작점 손실 방지
    pre_roll_buffer = collections.deque(maxlen=PRE_ROLL_FRAMES)
    current_utt = bytearray()  # 현재 발화의 원시 PCM 바이트 누적

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if not data:
                break

            # 1. 프리-롤 버퍼 갱신 (매 프레임마다 저장)
            pre_roll_buffer.append(data)

            # 2. 에너지 및 노이즈 플로어 계산
            e = rms_from_int16(data)
            if e < noise_floor * 1.5:
                noise_floor = (1 - EMA_ALPHA) * noise_floor + EMA_ALPHA * e

            # 3. VAD 판정 (에너지 게이트 + WebRTC VAD)
            is_loud_enough = e > (noise_floor * ENERGY_MARGIN)
            is_speech = is_loud_enough and vad.is_speech(data, RATE)

            # 4. 다수결 갱신
            votes.append(1 if is_speech else 0)
            speech_count = sum(votes)

            if not triggered:
                # 🌟 시작 트리거 (프리-롤 버퍼 추가)
                if speech_count >= WIN_MIN_VOICE:
                    triggered = True
                    hang = 0
                    current_utt.clear()
                    # 🌟 프리-롤 버퍼의 데이터를 먼저 추가하여 시작점 보정
                    current_utt.extend(b"".join(pre_roll_buffer))
                    print(f"\n▶ 발화 시작 (프리-롤 {PRE_ROLL_FRAMES}프레임 포함)")
            else:
                # 5. 종료 감지(행오버)
                if is_speech:
                    hang = 0
                else:
                    hang += 1

                if hang >= HANGOVER_FRAMES:
                    # 발화 종료 → STT로 보낼 준비
                    print(f"\n■ 발화 종료 (길이: {len(current_utt) / (RATE*2):.2f}초)")
                    if len(current_utt) > 0 and whisper_model:
                        with tempfile.TemporaryDirectory() as td:
                            wav_path = Path(td) / "utt.wav"
                            write_wav_int16(wav_path, bytes(current_utt), channels=CHANNELS, rate=RATE)
                            # === STT 호출 (로드된 모델 사용) ===
                            text = transcribe_with_whisper(whisper_model, str(wav_path), language="ko")
                            print(f"🗣️ 인식 결과: {text}")

                    triggered = False
                    hang = 0
                    current_utt.clear()

            # 6. 데이터 누적 (발화 중일 때)
            if triggered:
                current_utt.extend(data)

            # 7. 로그 출력
            if is_speech:
                print("🚨 음성 감지 (rms={:.4f}, floor={:.4f}, ratio={:.2f}x)".format(
                    e, noise_floor, e / max(noise_floor, EPS)), end='\r')
            else:
                print("… 침묵/잡음 (rms={:.4f}, floor={:.4f}, ratio={:.2f}x)".format(
                    e, noise_floor, e / max(noise_floor, EPS)), end='\r')

    except KeyboardInterrupt:
        print("\n녹음을 종료합니다.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("프로그램이 종료되었습니다.")


if __name__ == '__main__':
    main()

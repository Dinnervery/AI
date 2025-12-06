from __future__ import annotations
import re, json, wave, tempfile, collections
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# ===== 오디오 / VAD / STT =====
import numpy as np
import webrtcvad
import pyaudio

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# ===== 로컬 LLM (Qwen2.5, GGUF via llama.cpp) =====
from llama_cpp import Llama

LOCAL_LLM_OK = True
QWEN_MODEL_PATH = r"C:/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

_LOCAL_LLM: Optional[Llama] = None

def get_local_llm():
    global _LOCAL_LLM
    if _LOCAL_LLM is None:
        print("🧠 로컬 Qwen2.5 모델 로드 중...")
        _LOCAL_LLM = Llama(
            model_path=QWEN_MODEL_PATH,
            n_ctx=4096,
            n_threads=8,
            n_batch=256,
            verbose=False
        )
    return _LOCAL_LLM

# ===== 지역/시간대 =====
try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    from datetime import timezone, timedelta as _td
    KST = timezone(_td(hours=9))

# ===== 설정 =====
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
CHUNK = FRAME_SIZE

EMA_ALPHA = 0.02
ENERGY_MARGIN = 10.0
INITIAL_NOISE_FLOOR = 0.003
EPS = 1e-12

WIN_SIZE = 7
WIN_MIN_VOICE = 4
HANGOVER_FRAMES = 25
PRE_ROLL_FRAMES = 5

# ===== 도메인 정의 =====
DINNERS = {
    "valentine": "Valentine",
    "valentine dinner": "Valentine",
    "발렌타인": "Valentine",
    "밸런타인": "Valentine",
    "프렌치": "French",
    "프랜치": "French",
    "french": "French",
    "english": "English",
    "잉글리시": "English",
    "영국식": "English",
    "champagne feast": "ChampagneFeast",
    "샴페인 축제": "ChampagneFeast",
    "샴페인 축제 디너": "ChampagneFeast",
}

STYLES = {
    "simple": "Simple",
    "심플": "Simple",
    "간단": "Simple",
    "grand": "Grand",
    "그랜드": "Grand",
    "deluxe": "Deluxe",
    "디럭스": "Deluxe",
}

ITEM_ALIASES = {
    "baguette": "BAGUETTE",
    "바게트": "BAGUETTE",
    "바게트빵": "BAGUETTE",
    "바케트": "BAGUETTE",
    "바케트빵": "BAGUETTE",
    "champagne": "CHAMPAGNE",
    "샴페인": "CHAMPAGNE",
    "coffee pot": "COFFEE_POT",
    "커피 포트": "COFFEE_POT",
    "커피포트": "COFFEE_POT",
    "와인": "WINE",
    "스테이크": "STEAK",
    "샐러드": "SALAD",
    "베이컨": "BACON",
    "빵": "BREAD",
    "스크램블": "SCRAMBLED_EGGS",
    "스크램블 에그": "SCRAMBLED_EGGS",
}

AFFIRM = ["네", "예", "좋아요", "좋습니다", "맞아요", "맞습니다", "그렇게", "응", "ㅇㅇ", "확인"]
DENY = ["아니요", "아니오", "싫어요", "아닙니다", "그만", "취소", "변경 없어", "없어요", "없어"]

# ===== 유틸 =====
NUM_PAT = re.compile(r"(-?\d+)")
DATE_PAT = re.compile(r"(?:(\d{1,2})\s*월\s*(\d{1,2})\s*일)")

def kst_today():
    return datetime.now(tz=KST).date()

def parse_delivery_date(text: str) -> Optional[str]:
    """사용자 발화에서 배송일 후보를 뽑아 ISO로 변환"""
    t = text.strip()
    now = kst_today()
    if "내일" in t:
        return (now + timedelta(days=1)).isoformat()
    if "모레" in t:
        return (now + timedelta(days=2)).isoformat()

    m = DATE_PAT.search(t)
    if m:
        month = int(m.group(1)); day = int(m.group(2))
        year = now.year
        if now.month == 12 and month == 1:
            year += 1
        try:
            return datetime(year, month, day, tzinfo=KST).date().isoformat()
        except Exception:
            return None
    return None


# ===== 상태/주문 =====
@dataclass
class Order:
    dinner: Optional[str] = None
    style: Optional[str] = None
    qty: Dict[str, int] = field(default_factory=dict)
    delivery_date: Optional[str] = None  # ISO
    address: Optional[str] = None
    card: Optional[str] = None

    def apply_defaults(self):
        """디너 선택 시 기본 수량 채움"""
        if not self.dinner: 
            return

        def ensure(k: str, n: int):
            if k not in self.qty:
                self.qty[k] = n

        if self.dinner == "ChampagneFeast":
            ensure("CHAMPAGNE", 1)
            ensure("BAGUETTE", 4)
            ensure("COFFEE_POT", 1)
            ensure("WINE", 1)
            ensure("STEAK", 2)
        elif self.dinner == "French":
            ensure("WINE", 1); ensure("SALAD", 1); ensure("STEAK", 1)
        elif self.dinner == "English":
            ensure("SCRAMBLED_EGGS", 1); ensure("BACON", 1); ensure("BREAD", 1); ensure("STEAK", 1)
        elif self.dinner == "Valentine":
            ensure("WINE", 1); ensure("STEAK", 1)

    def summary_ko(self) -> str:
        parts = []
        if self.dinner:
            dko = {
                "Valentine":"발렌타인 디너",
                "French":"프렌치 디너",
                "English":"잉글리시 디너",
                "ChampagneFeast":"샴페인 축제 디너"
            }.get(self.dinner, self.dinner)
            parts.append(f"디너는 {dko}")
        if self.style:
            sko = {"Simple":"심플 스타일","Grand":"그랜드 스타일","Deluxe":"디럭스 스타일"}.get(self.style, self.style)
            parts.append(f"서빙은 {sko}")
        if self.qty:
            name_ko = {
                "BAGUETTE":"바게트빵","CHAMPAGNE":"샴페인","COFFEE_POT":"커피 포트",
                "WINE":"와인","STEAK":"스테이크","SALAD":"샐러드","BACON":"베이컨",
                "BREAD":"빵","SCRAMBLED_EGGS":"스크램블 에그"
            }
            qtxt = [f"{name_ko.get(k,k)} {v}개" for k,v in self.qty.items()]
            parts.append(", ".join(qtxt))
        return ", ".join(parts)

class State:
    START = "START"
    RUNNING = "RUNNING"
    ASK_DELIVERY_DATE = "ASK_DELIVERY_DATE"
    END = "END"


# ===== LLM 프롬프트/호출 =====
SYSTEM_PROMPT = r"""
너는 사용자 요구 기반의 디너 배달 상담원이다. 한국어로 공손하고 간결하게 대화한다.
출력은 반드시 JSON 하나의 객체로만 생성한다. keys: reply_ko, actions, missing_info.

이전 messages에는 사용자/시스템의 대화 기록이 포함될 수 있다. 
너는 기록을 참고해 이미 받은 정보는 반복해서 묻지 말고 다음 단계로 진행하라.

[허용값]
- dinner: Valentine, French, English, ChampagneFeast
- style: Simple, Grand, Deluxe
- item: BAGUETTE, CHAMPAGNE, COFFEE_POT, WINE, STEAK, SALAD, BACON, BREAD, SCRAMBLED_EGGS

[액션 스키마]
- SelectDinner(value)
- SelectStyle(value)
- ChangeQuantity(item, value)  # value는 최종 수량(절대치)
- SetDeliveryDate(value: YYYY-MM-DD)

[디너 설명]
- Valentine dinner: 하트/큐피드 장식 접시, 냅킨, 와인, 스테이크
- French dinner: 커피 1잔, 와인 1잔, 샐러드, 스테이크
- English dinner: 에그 스크램블, 베이컨, 빵, 스테이크
- Champagne Feast dinner: 항상 2인, 샴페인 1병, 바게트빵 4개, 커피 1포트, 와인, 스테이크

[스타일 설명]
- Simple: 플라스틱 접시/컵, 종이 냅킨, 플라스틱 쟁반 (와인 포함 시 플라스틱 잔)
- Grand: 도자기 접시/컵, 흰색 면 냅킨, 나무 쟁반 (와인 포함 시 플라스틱 잔)
- Deluxe: 꽃병, 도자기 접시/컵, 린넨 냅킨, 나무 쟁반 (와인 포함 시 유리 잔)

[대화 목표 순서]
1. dinner 확정
2. style 확정
3. quantity 확정/변경
4. deliveryDate 확정
5. 결제 안내 및 주문 확정 멘트

[규칙]
1) 사용자가 추천을 요청하면: 사유/상황을 한 문장으로 먼저 묻고, 사용자의 답을 받은 뒤에 해당 응답과 어울리는 dinner 종류를 제안.
2) dinner가 확정되면 style을 묻거나 제안.
3) dinner/style 관련 질문이 들어오면 위 정의 정보만 이용해 답변. 불필요한 추가 설명 금지.
4) 수량 변경 액션은 항상 최종 수량(절대치)로 출력한다.
5) missing_info는 아래 중 정확히 하나 또는 빈 배열만 허용:
   ["dinner"] | ["style"] | ["quantity"] | ["deliveryDate"] | []
6) 허용된 값 이외는 출력 금지.
7) 반드시 JSON 객체만 출력하고 다른 텍스트는 절대 포함하지 마라.
8) 같은 질문 재질문 금지

[출력 예시]
{
  "reply_ko": "어떤 기념일이신가요?",
  "actions": [],
  "missing_info": ["dinner"]
}
"""

def llm_decide(user_text: str, order: Order, history: List[dict], max_retries: int = 2) -> Optional[dict]:
    """LLM이 매 턴 reply/actions/missing_info를 결정. JSON 깨지면 재시도."""
    if not LOCAL_LLM_OK:
        return None

    llm = get_local_llm()

    ctx = {
        "dinner": order.dinner,
        "style": order.style,
        "qty": order.qty,
        "delivery_date": order.delivery_date,
    }

    user_payload = json.dumps(
        {"user_text": user_text, "context": ctx},
        ensure_ascii=False
    )

    messages_base = [
        {"role":"system", "content": SYSTEM_PROMPT},
        *history, 
        {"role":"user", "content": user_payload}
    ]

    ok_enums = {
        "dinners": {"Valentine","French","English","ChampagneFeast"},
        "styles": {"Simple","Grand","Deluxe"},
        "items": {"BAGUETTE","CHAMPAGNE","COFFEE_POT","WINE","STEAK","SALAD","BACON","BREAD","SCRAMBLED_EGGS"},
        "missing": {"dinner","style","quantity","deliveryDate"},
    }

    for attempt in range(max_retries + 1):
        try:
            resp = llm.create_chat_completion(
                messages=messages_base if attempt == 0 else (
                    messages_base + [{
                        "role":"system",
                        "content":"이전 답변이 JSON 규칙을 위반했다. 반드시 JSON 객체 하나만, 허용값만 사용해 다시 출력하라."
                    }]
                ),
                temperature=0.1,
                max_tokens=256
            )

            txt = resp["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", txt, flags=re.S)
            if not m:
                continue

            data = json.loads(m.group(0))
            if not isinstance(data, dict):
                continue
            if "actions" not in data or "reply_ko" not in data or "missing_info" not in data:
                continue

            # ----- actions 클리닝 -----
            cleaned_actions = []
            for a in data.get("actions", []):
                if not isinstance(a, dict):
                    continue
                t = a.get("type")

                if t == "SelectDinner" and a.get("value") in ok_enums["dinners"]:
                    cleaned_actions.append({"type":"SelectDinner","value":a["value"]})

                elif t == "SelectStyle" and a.get("value") in ok_enums["styles"]:
                    cleaned_actions.append({"type":"SelectStyle","value":a["value"]})

                elif t == "ChangeQuantity" and a.get("item") in ok_enums["items"]:
                    try:
                        val = int(a.get("value"))
                        cleaned_actions.append({"type":"ChangeQuantity","item":a["item"],"value":val})
                    except Exception:
                        pass

                elif t == "SetDeliveryDate":
                    v = str(a.get("value"))
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                        cleaned_actions.append({"type":"SetDeliveryDate","value":v})

            data["actions"] = cleaned_actions

            # ----- missing_info 표준화 -----
            miss = data.get("missing_info", [])
            if not isinstance(miss, list):
                miss = []
            miss = [x for x in miss if x in ok_enums["missing"]]
            if len(miss) > 1:
                miss = miss[:1]
            data["missing_info"] = miss

            return data

        except Exception:
            continue

    return None


# ===== 대화 관리자 (LLM-only 대화, 룰은 안전장치만) =====
class DialogueManager:
    def __init__(self, use_llm: bool = True):
        self.state = State.START
        self.order = Order()
        self.use_llm = use_llm and LOCAL_LLM_OK
        self._clicked = False
        self.history: List[dict] = []  
        self.history_max = 8   

    def emit(self, action: str):
        print(f"[ACTION] {action}")
        if action.strip().startswith("ClickOrder("):
            raise KeyboardInterrupt

    def say(self, msg: str):
        print(f"시스템: {msg}")

    def _maybe_click_order(self):
        if not self._clicked:
            self.emit("ClickOrder()")
            self._clicked = True

    def handle_user(self, user_text: str):
        user_text = (user_text or "").strip()

        # START 진입 시: 첫 턴도 LLM이 자연스럽게 시작하도록 트리거 문장 제공
        if self.state == State.START:
            self.state = State.RUNNING
            user_text = user_text or "대화를 시작합니다."

        # 사용자가 말한 배송일 후보는 코드에서도 미리 잡아 context를 강화
        maybe_date = parse_delivery_date(user_text)
        if maybe_date:
            self.order.delivery_date = maybe_date

        if not self.use_llm:
            self.say("현재 LLM이 비활성화돼 있습니다. 설정을 확인해 주세요.")
            return

        # ✅ 매 턴 LLM 호출(대화 기록 포함)
        data = llm_decide(user_text, self.order, self.history)
        if not data:
            # LLM이 JSON을 계속 못 내면 유도 멘트
            self.say("입력을 잘 이해하지 못했습니다. 조금 더 구체적으로 다시 말씀해 주세요.")
            return

        # ---- LLM 결정 적용 (reply 출력 + actions 반영 + 최소 상태 힌트) ----
        self._apply_llm_decision(data)

        # ✅ 히스토리 누적: user → assistant
        reply = data.get("reply_ko") or ""
        self.history.append({"role": "user", "content": user_text})
        if reply:
            self.history.append({"role": "assistant", "content": reply})

        # ✅ 히스토리 길이 제한(최근 N개 메시지만 유지)
        if len(self.history) > self.history_max:
            self.history = self.history[-self.history_max:]

        # ✅ 주문 종료 조건: 핵심 정보 모두 있고, 더 물을 게 없다고 LLM이 판단했을 때
        if (self.order.dinner and self.order.style and self.order.delivery_date
            and data.get("missing_info") == []):
            self.say(self.final_confirmation())
            self._maybe_click_order()
            self.state = State.END


    def _apply_llm_decision(self, data: dict):
        reply = data.get("reply_ko") or ""
        acts: List[dict] = data.get("actions", [])
        miss: List[str] = data.get("missing_info", [])

        # 1) LLM 자연어 응답 출력
        if reply:
            self.say(reply)

        # 2) 액션 반영(검증된 것만 들어옴)
        for a in acts:
            t = a.get("type")
            if t == "SelectDinner":
                v = a.get("value")
                if v:
                    self.order.dinner = v
                    self.order.apply_defaults()
                    self.emit(f"SelectDinner({v})")

            elif t == "SelectStyle":
                v = a.get("value")
                if v:
                    self.order.style = v
                    self.emit(f"SelectStyle({v})")

            elif t == "ChangeQuantity":
                it = a.get("item")
                v = int(a.get("value", 0))
                self.order.qty[it] = max(0, v)
                self.emit(f"ChangeQuantity({it}, {v})")

            elif t == "SetDeliveryDate":
                v = a.get("value")
                if v:
                    self.order.delivery_date = v
                    self.emit(f"SetDeliveryDate({v})")

        # 3) missing_info 기반 상태 힌트(대화는 LLM이 알아서 하므로 상태는 최소만)
        if "deliveryDate" in miss:
            self.state = State.ASK_DELIVERY_DATE
        else:
            self.state = State.RUNNING

    def final_confirmation(self) -> str:
        if self.order.delivery_date:
            y,m,d = self.order.delivery_date.split("-")
            dtxt = f"{int(m)}월 {int(d)}일"
        else:
            dtxt = "지정하신 날짜"
        return f"{self.order.summary_ko()}로 주문 접수하겠습니다. 배송일은 {dtxt}로 진행하겠습니다. 감사합니다."


# ===== STT 유틸 =====
def rms_from_int16(pcm_bytes: bytes) -> float:
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x) + EPS))

def write_wav_int16(path: Path, pcm_bytes: bytes, channels=1, rate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)

# ===== Whisper 모델 =====
_WHISPER_MODEL: Optional[WhisperModel] = None

def get_whisper_model():
    global _WHISPER_MODEL
    if not WHISPER_AVAILABLE:
        print("[경고] faster-whisper 미설치. STT 비활성화.")
        return None
    if _WHISPER_MODEL is None:
        print("🧠 Whisper 모델 로드 중... (medium, int8, CPU)")
        _WHISPER_MODEL = WhisperModel("medium", device="cpu", compute_type="int8")
    return _WHISPER_MODEL

def transcribe_with_whisper(model: WhisperModel, wav_path: str, language="ko") -> str:
    if model is None:
        return ""
    segs, info = model.transcribe(wav_path, language=language, vad_filter=False)
    return "".join(s.text for s in segs).strip()

# ===== 메인 루프 =====
DM = DialogueManager(use_llm=True)

def on_transcript(text: str):
    if not text:
        return
    print(f"고객: {text}")
    DM.handle_user(text)

def run_loop():
    vad = webrtcvad.Vad(3)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

    # Whisper 선로딩
    model = get_whisper_model()

    # Qwen 선로딩 + 워밍업
    if DM.use_llm:
        llm = get_local_llm()
        try:
            llm.create_chat_completion(
                messages=[
                    {"role":"system","content":"테스트. JSON만 출력해라."},
                    {"role":"user","content":'{"user_text":"테스트","context":{}}'}
                ],
                temperature=0.0,
                max_tokens=8
            )
        except Exception:
            pass

    print("마이크 녹음을 시작합니다... Ctrl+C 로 종료.\n")
    DM.handle_user("")  # START 트리거

    noise_floor = INITIAL_NOISE_FLOOR
    votes = collections.deque(maxlen=WIN_SIZE)
    triggered = False
    hang = 0
    pre_roll = collections.deque(maxlen=PRE_ROLL_FRAMES)
    utt = bytearray()

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not data:
                break
            pre_roll.append(data)

            e = rms_from_int16(data)
            if e < noise_floor * 1.5:
                noise_floor = (1-EMA_ALPHA)*noise_floor + EMA_ALPHA*e

            is_loud = e > (noise_floor * ENERGY_MARGIN)
            is_speech = is_loud and vad.is_speech(data, RATE)

            votes.append(1 if is_speech else 0)
            speech_cnt = sum(votes)

            if not triggered:
                if speech_cnt >= WIN_MIN_VOICE:
                    triggered = True; hang = 0; utt.clear()
                    utt.extend(b"".join(pre_roll))
                    print("\n▶ 발화 시작")
            else:
                if is_speech:
                    hang = 0
                else:
                    hang += 1
                if hang >= HANGOVER_FRAMES:
                    print("\n■ 발화 종료")
                    if len(utt) > 0 and model is not None:
                        with tempfile.TemporaryDirectory() as td:
                            wav_path = Path(td)/"utt.wav"
                            write_wav_int16(wav_path, bytes(utt), channels=CHANNELS, rate=RATE)
                            text = transcribe_with_whisper(model, str(wav_path), language="ko")
                            if text:
                                on_transcript(text)
                    triggered = False; hang = 0; utt.clear()

            if triggered:
                utt.extend(data)

    except KeyboardInterrupt:
        print("\n종료를 수행합니다.")
    finally:
        stream.stop_stream(); stream.close(); audio.terminate()
        print("프로그램이 종료되었습니다.")

def run_text_loop():
    print("키보드 입력 모드입니다. 종료하려면 '종료' 또는 'quit' 입력.\n")
    DM.handle_user("")  # START 트리거 (LLM이 첫 인사 만들게)

    while True:
        try:
            text = input("고객(키보드): ").strip()
        except EOFError:
            break

        if text in ("종료", "quit", "exit"):
            print("프로그램을 종료합니다.")
            break

        if text:
            DM.handle_user(text)

if __name__ == "__main__":
    run_text_loop()

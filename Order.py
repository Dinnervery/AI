from __future__ import annotations
import os, re, json, wave, tempfile, collections
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

# ===== 오디오 / VAD / STT =====
import numpy as np
import webrtcvad
import pyaudio

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# ===== LLM (선택) =====
OPENAI_OK = False
try:
    from openai import OpenAI
    _ = os.environ.get("OPENAI_API_KEY")
    OPENAI_OK = bool(_)
except Exception:
    OPENAI_OK = False

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

def extract_item_and_value(text: str) -> Optional[Tuple[str, int, bool]]:
    """아이템, 값, is_absolute(절대치 여부)을 추출.
       예: "바게트 6개로" → (BAGUETTE, 6, True)
           "샴페인 2병 더" → (CHAMPAGNE, 2, False)
    """
    t = text.lower().strip()
    item_key = None
    for k, v in ITEM_ALIASES.items():
        if k in t:
            item_key = v; break
    if not item_key:
        return None
    m = NUM_PAT.search(t)
    if not m:
        return None
    n = int(m.group(1))
    abs_flag = ("로" in t or "으로" in t or "변경" in t)
    if ("더" in t or "+" in t or "추가" in t) and not abs_flag:
        return (item_key, n, False)
    if ("빼" in t or "줄" in t or "-" in t) and not abs_flag:
        return (item_key, -n, False)
    return (item_key, n, True)

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
        if not self.dinner: return
        def ensure(k: str, n: int):
            if k not in self.qty: self.qty[k] = n
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
            dko = {"Valentine":"발렌타인 디너","French":"프렌치 디너","English":"잉글리시 디너","ChampagneFeast":"샴페인 축제 디너"}.get(self.dinner, self.dinner)
            parts.append(f"디너는 {dko}")
        if self.style:
            sko = {"Simple":"심플 스타일","Grand":"그랜드 스타일","Deluxe":"디럭스 스타일"}.get(self.style, self.style)
            parts.append(f"서빙은 {sko}")
        if self.qty:
            name_ko = {"BAGUETTE":"바게트빵","CHAMPAGNE":"샴페인","COFFEE_POT":"커피 포트","WINE":"와인","STEAK":"스테이크","SALAD":"샐러드","BACON":"베이컨","BREAD":"빵","SCRAMBLED_EGGS":"스크램블 에그"}
            qtxt = [f"{name_ko.get(k,k)} {v}개" for k,v in self.qty.items()]
            parts.append(", ".join(qtxt))
        return ", ".join(parts)

class State:
    START = "START"
    ASK_OCCASION = "ASK_OCCASION"
    GOT_DINNER = "GOT_DINNER"
    SUMMARY = "SUMMARY"
    ASK_QUANTITY = "ASK_QUANTITY"
    ASK_ANYTHING_ELSE = "ASK_ANYTHING_ELSE"
    ASK_DELIVERY_DATE = "ASK_DELIVERY_DATE"
    END = "END"

# ===== LLM 프롬프트/호출 =====
SYSTEM_PROMPT = (
    "너는 사용자 요구 기반의 디너 배달 상담원이다. 한국어로 공손하고 간결하게 대화한다. "
    "출력은 반드시 JSON 하나의 객체로만 생성한다. keys: reply_ko, actions, missing_info.\n"
    "[디너] Valentine, French, English, ChampagneFeast 만 허용.\n"
    "[스타일] Simple, Grand, Deluxe 만 허용.\n"
    "[아이템] BAGUETTE, CHAMPAGNE, COFFEE_POT, WINE, STEAK, SALAD, BACON, BREAD, SCRAMBLED_EGGS 만 허용.\n"
    "[액션] SelectDinner(value), SelectStyle(value), ChangeQuantity(item,value), SetDeliveryDate(value: YYYY-MM-DD).\n"
    "규칙: 1) 추천 요청엔 한 문장으로 사유 묻고 2~3개 제안. 2) 메뉴 정해지면 스타일 제안. 3) 메뉴/스타일 정해지면 요약 후 수량 변경 여부 질문. "
    "4) 수량 변경은 가급적 절대치(value=정수)로. 5) 누락 정보는 missing_info 배열에 이름을 넣고 reply_ko로 한 번에 하나만 물어본다. 6) 허용된 값 이외는 사용 금지."
)

def llm_decide(user_text: str, order: Order) -> Optional[dict]:
    if not OPENAI_OK:
        return None
    try:
        client = OpenAI()
        # 컨텍스트를 축약해 전달(핵심 슬롯만)
        ctx = {
            "dinner": order.dinner,
            "style": order.style,
            "qty": order.qty,
            "delivery_date": order.delivery_date,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"user_text": user_text, "context": ctx}, ensure_ascii=False)}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            response_format={"type":"json_object"},
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        # 간단 검증
        if not isinstance(data, dict):
            return None
        if "actions" not in data or "reply_ko" not in data:
            return None
        # 액션 값 정합성 대략 체크
        ok_enums = {
            "dinners": {"Valentine","French","English","ChampagneFeast"},
            "styles": {"Simple","Grand","Deluxe"},
            "items": {"BAGUETTE","CHAMPAGNE","COFFEE_POT","WINE","STEAK","SALAD","BACON","BREAD","SCRAMBLED_EGGS"},
        }
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
        return data
    except Exception:
        return None

# ===== 규칙 라우터 =====

def is_recommend_request(text: str) -> bool:
    t = text.strip()
    return any(kw in t for kw in ["추천", "뭐가 좋아", "골라줘", "추천해"])

def detect_dinner(text: str) -> Optional[str]:
    t = text.lower()
    for k, v in DINNERS.items():
        if k in t: return v
    return None

def detect_style(text: str) -> Optional[str]:
    t = text.lower()
    for k, v in STYLES.items():
        if k in t: return v
    return None

def is_affirm(text: str) -> bool:
    return any(a in text for a in AFFIRM)

def is_deny(text: str) -> bool:
    return any(a in text for a in DENY)

# ===== 대화 관리자 =====
class DialogueManager:
    def __init__(self, use_llm: bool = True):
        self.state = State.START
        self.order = Order()
        self.use_llm = use_llm and OPENAI_OK

    # 인터페이스
    def emit(self, action: str):
        print(f"[ACTION] {action}")

    def say(self, msg: str):
        print(f"시스템: {msg}")

    # 메인 핸들러
    def handle_user(self, user_text: str):
        user_text = user_text.strip()
        if self.state == State.START:
            self.say("안녕하세요, 어떤 디너를 주문하시겠습니까? 추천이 필요하시면 말씀해 주세요.")
            self.state = State.ASK_OCCASION
            return

        # 언제든 날짜가 튀어나오면 캐치
        maybe_date = parse_delivery_date(user_text)
        if maybe_date:
            self.order.delivery_date = maybe_date

        # LLM 라우팅 조건: 모호/복합/추천/서술형 문장
        go_llm = self.use_llm and (
            is_recommend_request(user_text)
            or (not detect_dinner(user_text) and not detect_style(user_text) and not extract_item_and_value(user_text) and not is_affirm(user_text) and not is_deny(user_text))
        )
        if go_llm:
            data = llm_decide(user_text, self.order)
            if data:
                self._apply_llm_decision(data)
                return
            # LLM 실패 시 규칙으로 폴백

        # ===== 규칙 경로 =====
        if self.state == State.ASK_OCCASION:
            d = detect_dinner(user_text)
            if d:
                self.order.dinner = d; self.order.apply_defaults()
                self.emit(f"SelectDinner({d})")
                self.say("스타일은 심플, 그랜드, 디럭스 중 어떤 것으로 하시겠습니까?")
                self.state = State.GOT_DINNER
                return
            if is_recommend_request(user_text):
                self.say("무슨 기념일이신지 한 마디로 알려주시면 2~3가지로 제안드리겠습니다.")
                return
            if any(k in user_text for k in ["생일","생신","기념일","프로포즈","졸업","승진","돌잔치","환갑"]):
                self.say("축하합니다. 프렌치 디너 또는 샴페인 축제 디너를 추천드립니다. 어떤 메뉴로 하시겠습니까?")
                return
            self.say("추천이 필요하신가요, 아니면 특정 디너를 정하셨나요?")
            return

        if self.state == State.GOT_DINNER:
            st = detect_style(user_text)
            if st:
                self.order.style = st
                self.emit(f"SelectStyle({st})")
                self.say(f"네, {self.order.summary_ko()}로 진행하겠습니다. 수량 변경이 필요하신가요?")
                self.state = State.SUMMARY
                return
            self.say("심플/그랜드/디럭스 중에서 선택 부탁드립니다.")
            return

        if self.state == State.SUMMARY:
            ch = extract_item_and_value(user_text)
            if ch:
                item, val, is_abs = ch
                if is_abs:
                    self.order.qty[item] = max(0, int(val))
                    self.emit(f"ChangeQuantity({item}, {int(val)})")
                else:
                    newv = max(0, self.order.qty.get(item, 0) + int(val))
                    delta = newv - self.order.qty.get(item, 0)
                    self.order.qty[item] = newv
                    self.emit(f"ChangeQuantity({item}, {delta:+d})")
                self.say(f"알겠습니다. 현재 요약: {self.order.summary_ko()}. 다른 변경이 필요하신가요?")
                self.state = State.ASK_QUANTITY
                return
            if is_affirm(user_text) or is_deny(user_text):
                self.say("추가로 필요하신 것이 있으실까요?")
                self.state = State.ASK_ANYTHING_ELSE
                return
            self.say("수량을 바꾸시려면 예: '바게트 6개로', '샴페인 2병으로'처럼 말씀해 주세요.")
            return

        if self.state == State.ASK_QUANTITY:
            ch = extract_item_and_value(user_text)
            if ch:
                item, val, is_abs = ch
                if is_abs:
                    self.order.qty[item] = max(0, int(val))
                    self.emit(f"ChangeQuantity({item}, {int(val)})")
                else:
                    newv = max(0, self.order.qty.get(item, 0) + int(val))
                    delta = newv - self.order.qty.get(item, 0)
                    self.order.qty[item] = newv
                    self.emit(f"ChangeQuantity({item}, {delta:+d})")
                self.say(f"네, 업데이트했습니다. 현재 요약: {self.order.summary_ko()}. 더 변경하시겠습니까?")
                return
            if is_affirm(user_text) or is_deny(user_text):
                self.say("추가로 필요하신 것이 있으실까요?")
                self.state = State.ASK_ANYTHING_ELSE
                return
            self.say("변경이 없으시면 '없어요'라고 말씀해 주세요.")
            return

        if self.state == State.ASK_ANYTHING_ELSE:
            if is_deny(user_text):
                if not self.order.delivery_date:
                    self.say("원하시는 배송일을 말씀해 주세요. 예: '12월 3일', '내일', '모레'")
                    self.state = State.ASK_DELIVERY_DATE
                    return
                else:
                    self.say(self.final_confirmation())
                    self.state = State.END
                    return
            # 추가 변경/선택
            ch = extract_item_and_value(user_text)
            if ch:
                item, val, is_abs = ch
                if is_abs:
                    self.order.qty[item] = max(0, int(val))
                    self.emit(f"ChangeQuantity({item}, {int(val)})")
                else:
                    newv = max(0, self.order.qty.get(item, 0) + int(val))
                    delta = newv - self.order.qty.get(item, 0)
                    self.order.qty[item] = newv
                    self.emit(f"ChangeQuantity({item}, {delta:+d})")
                self.say(f"반영했습니다. 현재 요약: {self.order.summary_ko()}. 더 필요하신가요?")
                return
            d = detect_dinner(user_text)
            if d:
                self.order.dinner = d; self.order.apply_defaults()
                self.emit(f"SelectDinner({d})")
                self.say("스타일은 심플/그랜드/디럭스 중에서 선택해 주세요.")
                self.state = State.GOT_DINNER
                return
            st = detect_style(user_text)
            if st:
                self.order.style = st; self.emit(f"SelectStyle({st})")
                self.say(f"업데이트했습니다. 현재 요약: {self.order.summary_ko()}.")
                return
            if maybe_date:
                self.say(self.final_confirmation())
                self.state = State.END
                return
            self.say("변경이 없으시면 '없어요'라고 말씀해 주세요. 변경이 있으면 항목과 수량을 말씀해 주세요.")
            return

        if self.state == State.ASK_DELIVERY_DATE:
            if maybe_date:
                self.say(self.final_confirmation())
                self.state = State.END
                return
            self.say("예: '12월 3일', '내일', '모레'처럼 말씀해 주세요.")
            return

        if self.state == State.END:
            self.say("주문을 마무리했습니다. 감사합니다.")
            return

    def _apply_llm_decision(self, data: dict):
        reply = data.get("reply_ko") or ""
        acts: List[dict] = data.get("actions", [])
        miss: List[str] = data.get("missing_info", [])

        # 자연어 멘트
        if reply:
            self.say(reply)
        # 액션 실행
        for a in acts:
            t = a.get("type")
            if t == "SelectDinner":
                v = a.get("value");
                if v:
                    self.order.dinner = v; self.order.apply_defaults()
                    self.emit(f"SelectDinner({v})")
            elif t == "SelectStyle":
                v = a.get("value");
                if v:
                    self.order.style = v
                    self.emit(f"SelectStyle({v})")
            elif t == "ChangeQuantity":
                it = a.get("item"); v = int(a.get("value", 0))
                self.order.qty[it] = max(0, int(v))
                self.emit(f"ChangeQuantity({it}, {int(v)})")
            elif t == "SetDeliveryDate":
                v = a.get("value");
                if v: self.order.delivery_date = v
        # 상태 전이(간단)
        if self.order.dinner and not self.order.style:
            self.state = State.GOT_DINNER
        elif self.order.dinner and self.order.style and not miss:
            # 메뉴/스타일이 정해졌다면 요약 단계로 유도
            self.state = State.SUMMARY
        # 누락 정보가 있다면 적절 질문 상태로
        if "deliveryDate" in miss and self.state != State.ASK_DELIVERY_DATE:
            self.state = State.ASK_DELIVERY_DATE

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
    if not text: return
    print(f"고객: {text}")
    DM.handle_user(text)


def run_loop():
    vad = webrtcvad.Vad(3)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    model = get_whisper_model()

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
            if not data: break
            pre_roll.append(data)

            # 에너지 추정
            e = rms_from_int16(data)
            if e < noise_floor * 1.5:
                noise_floor = (1-EMA_ALPHA)*noise_floor + EMA_ALPHA*e

            is_loud = e > (noise_floor * ENERGY_MARGIN)
            is_speech = is_loud and vad.is_speech(data, RATE)

            votes.append(1 if is_speech else 0)
            speech_cnt = sum(votes)

            if not triggered:
                if speech_cnt >= WIN_MIN_VOICE:
                    triggered = True; hang = 0; utt.clear();
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

if __name__ == "__main__":
    run_loop()

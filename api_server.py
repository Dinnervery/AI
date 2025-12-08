# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from Order_groq import DialogueManager

app = FastAPI()

# FastAPI 예시
@app.get("/health")
def health():
    return {"status": "healthy"}

# CORS: 나중에 웹/프론트에서 호출할 수 있게 열어두기
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요하면 도메인 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 대화 매니저 (서버가 살아있는 동안 상태 유지)
dm = DialogueManager(use_llm=True)

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    reply: str
    state: str
    actions: list
    order_summary: str | None = None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = dm.handle_user(req.text)

    return ChatResponse(
        reply=result["reply"],
        state=dm.state,
        actions=result["actions"],
        order_summary=result["order_summary"],
    )

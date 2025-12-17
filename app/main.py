from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- IMPORTANTE! ele serve pra poder usar em navegadores
from pydantic import BaseModel
from typing import List, Optional
from app.services.llm_service import SupportAgent

app = FastAPI(title="AI Support Agent API", version="1.0.0")

# --- CONFIGURAÇÃO DE CORS (O Segredo) ---
# Isso permite que seu Front-end converse com seu Back-end
origins = [
    "http://localhost:3000",          # Para seus testes locais
    "https://seu-portfolio.vercel.app", # URL do seu portfólio (troque pela real)
    "*"                               # Em último caso, use "*" para liberar para todos (bom para testes)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Inicializa o Agente
agent = SupportAgent()

# Modelo de dados (O que a API espera receber)
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = [] # Lista de {role: "", content: ""}


@app.get("/")
async def root():
    return {"message": "O Agente de Suporte está Online! Acesse /docs para usar."}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal para conversar com o Agente de Suporte.
    Recebe a mensagem atual e o histórico da conversa.
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")
    
    response_text = agent.get_response(request.message, request.history)
    
    return {"response": response_text}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "AI Support Agent Running"}
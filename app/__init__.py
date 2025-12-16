from fastapi import FastAPI
from .services.llm_service import llm_router # Você provavelmente precisará disto

# 1. Crie a instância principal da aplicação (o segundo 'app' no gunicorn app:app)
app = FastAPI(
    title="AI Support Agent",
    description="Agente de suporte utilizando LLM.",
    version="1.0.0"
)

# 2. Inclua suas rotas (endpoints)
# Se suas rotas estiverem em app/services/llm_service.py (como é comum), adicione:
app.include_router(llm_router, prefix="/api/v1")

# 3. Adicione uma rota básica para teste (opcional, mas recomendado)
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}

# Lembre-se de que você provavelmente precisará ter o objeto 'llm_router'
# definido e exportado no arquivo 'app/services/llm_service.py'.
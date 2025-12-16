import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Carrega ambiente
load_dotenv()

class SupportAgent:
    def __init__(self):
        token = os.getenv("HF_API_KEY")
        # Configuração do Cliente e do Prompt do Sistema
        self.client = InferenceClient(model="microsoft/Phi-3-mini-4k-instruct", token=token)
        self.system_prompt = (
            "Você é um Assistente de Suporte Técnico extremamente focado e eficiente. "
            "Sua função é fornecer a solução técnica exata. "
            "Sua resposta deve começar com 'Solução:'."
        )

    def _sanitize_input(self, user_input: str) -> str:
        blocklist = ["ignore as instruções", "developer mode", "jailbreak"]
        sanitized = user_input.lower()
        for word in blocklist:
            if word in sanitized:
                return "##INPUT_BLOCKED##"
        return user_input

    def get_response(self, message: str, history: list) -> str:
        # 1. Sanitização
        safe_input = self._sanitize_input(message)
        if safe_input == "##INPUT_BLOCKED##":
            return "Alerta de Segurança: Seu input contém instruções não permitidas."

        # 2. Montagem do Contexto
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history) # Adiciona histórico anterior
        messages.append({"role": "user", "content": safe_input})

        # 3. Chamada ao LLM
        try:
            response = self.client.chat_completion(messages, max_tokens=500, stream=False)
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro interno no processamento do LLM: {str(e)}"
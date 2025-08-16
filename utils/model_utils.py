import asyncio
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Ensure Windows + Streamlit threads always have an event loop
def _get_or_create_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:  # if no loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def get_embedding_model(model: str = "models/embedding-001"):
    _get_or_create_loop()  # 👈 ensure loop exists
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

def get_llm_model(model: str = "gemini-1.5-flash"):
    _get_or_create_loop()  # 👈 ensure loop exists
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

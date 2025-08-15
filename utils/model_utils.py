from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def get_embedding_model(model: str = "models/embedding-001"):
    return GoogleGenerativeAIEmbeddings(model=model)

def get_llm_model(model = "gemini-1.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model, 
        temperature=0.3
    )
    return llm
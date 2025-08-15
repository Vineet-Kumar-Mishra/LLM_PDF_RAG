from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

DB_LOCAL = config["VECTOR_DB_NAME"]


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text(pdf_file: str):
    loader = PyMuPDFLoader(pdf_file)
    text = loader.load()
    return text


def chunk_docs(text, chunk_size: int, overlap: int):
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = overlap)
    chunks = textsplitter.split_documents(text)
    return chunks

def get_embedding_model(model: str = "models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model=model)
    return embeddings


def get_vector_store(chunks, embedding_model):
    if os.path.exists(DB_LOCAL):
        vectordb = FAISS.load_local(DB_LOCAL, embedding_model, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(chunks, embedding_model)
        vectordb.save_local(DB_LOCAL)
    return vectordb
    
def get_llm_chain(llm):
    prompt_template = """
    You are an Indian Polity subject specialist assistant.
    Answer the questions using the provided context.
    If you don't know the answer, say "I don't know!"
    
    Context:
    {context}
    
    Conversation history:
    {chat_history}
    
    Question:
    {question}
    
    Answer:
    """
    QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
        )
    chain = load_qa_chain(llm, chain_type="stuff",prompt=QA_PROMPT)

    return chain

def user_input_preprocess(query, vector_db, chain):
    docs = vector_db.similarity_search(query)

    response = chain.invoke(
        {
            "input_documents": docs,
            "question": query,
            "chat_history": ""
        },
        return_only_outputs=True
    )
    return response

text = extract_text(".\pdf_files\polity.pdf")

chunks = chunk_docs(text, 250, 50)

embedding_model = get_embedding_model()

vector_db = get_vector_store(chunks, embedding_model)

#"gemini-1.5-flash", "gemini-1.5-pro"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                            temperature=0.3)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chain = get_llm_chain(llm)

response = user_input_preprocess("What is supreme court",vector_db, chain)

print(response["output_text"])
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text(pdf_file):
    if isinstance(pdf_file, str):
        pdf_file = [pdf_file]
    all_docs = []
    for file in pdf_file:
        loader = PyMuPDFLoader(file)
        docs = loader.load()  
        all_docs.extend(docs)
    return all_docs


def chunk_docs(documents, chunk_size: int, overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks


def get_prompt():
    prompt_template = """You are a helpful conversational assistant.
    If the user’s question is related to the provided context, use that context to answer.
    If it’s unrelated, respond naturally without requiring the context.

    Context (optional):
    {context}

    Conversation history:
    {chat_history}

    User:
    {question}

    Assistant:

    """
    return prompt_template

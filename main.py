from utils.text_utils import extract_text, chunk_docs, get_prompt
from utils.model_utils import get_embedding_model, get_llm_model
from utils.vector_utils import get_vector_store
from utils.chain_utils import enable_memory, get_llm_chain, summarise_document, qa_mode, free_chat, detect_mode
from dotenv import load_dotenv
import glob

import yaml

load_dotenv()

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# vector DB----------------------------------------
DB_LOCAL = config["VECTOR_DB_NAME"]
CREATE_NEW_INDEX = config["CREATE_NEW_INDEX"]

# Models-------------------------------------------
LLM_MODEL = config["LLM_MODEL"]
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]

# Mode selection-----------------------------------
MODE = config["MODE"]
MODE_SELECTOR = config["MODE_SELECTOR"]

# PDF files-----------------------------------------
PDF_FILES = config["PDF_FILES"]
all_pdf_files = glob.glob(PDF_FILES+"/*pdf")

if __name__=="__main__":
    text_docs = extract_text(all_pdf_files)
    chunks = chunk_docs(text_docs, chunk_size=250, overlap=50)

    embedding_model = get_embedding_model(model=EMBEDDING_MODEL)

    vector_db = get_vector_store(chunks, embedding_model,DB_LOCAL,CREATE_NEW_INDEX)

    llm = get_llm_model(model = LLM_MODEL)
    memory = enable_memory()
    chain = get_llm_chain(llm)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        if MODE_SELECTOR=='LLM':
            mode = detect_mode(query, llm)
        else:
            mode = MODE

        if mode == "summary":
            answer = summarise_document(text_docs, llm)
        elif mode == "qa":
            answer = qa_mode(query, vector_db, chain, memory)
        else:
            answer = free_chat(query, llm, memory)


        print(f"AI: {answer}")
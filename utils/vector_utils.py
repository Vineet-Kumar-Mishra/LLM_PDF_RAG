from langchain.vectorstores import FAISS
import os

def get_vector_store(chunks, embedding_model,DB_LOCAL,CREATE_NEW_INDEX):

    if os.path.exists(DB_LOCAL) and CREATE_NEW_INDEX==False:
        vectordb = FAISS.load_local(DB_LOCAL, embedding_model, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(chunks, embedding_model)
        vectordb.save_local(DB_LOCAL)
    return vectordb
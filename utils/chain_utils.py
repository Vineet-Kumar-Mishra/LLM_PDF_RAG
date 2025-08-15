from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from utils.text_utils import get_prompt

def enable_memory():
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True
    )
    return memory

def get_llm_chain(llm):
    prompt_template = get_prompt()
    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)
    return chain

def summarise_document(documents, llm):
    full_text = "\n".join(doc.page_content for doc in documents[:20])  # limit for safety
    summary_prompt = f"Summarise the following text in a clear and concise manner:\n\n{full_text}"
    response = llm.invoke(summary_prompt)
    return response.content

def qa_mode(query, vector_db, chain, memory):
    docs = vector_db.similarity_search(query, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    response = chain.invoke({
        "input_documents": docs,
        "context": context_text,
        "question": query,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })

    # Save query + answer to memory
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response["output_text"])
    return response["output_text"]

def free_chat(query, llm, memory):
    chat_prompt = f"""
    Conversation so far: {memory.load_memory_variables({})["chat_history"]}
    User: {query}
    """
    response = llm.invoke(chat_prompt)
    return response.content

def detect_mode(query, llm):
    classification_prompt = f"""
    You are a mode classification assistant.

    Modes:
    1. "summary" → if the user asks to summarise the PDF/document.
    2. "qa" → if the user asks a question about the PDF/document's content.
    3. "chat" → if the user is just having a general conversation unrelated to the PDF.

    User query: "{query}"

    Reply with only one word: summary, qa, or chat.
    """
    response = llm.invoke(classification_prompt)
    return response.content.strip().lower()
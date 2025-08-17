from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
from chroma_utils import vectorstore
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("LLM_for_chat")

# ───────────────────────── Prompts ──────────────────────────
contextualize_q_system = (
    """You are CHOTU, a helpful AI assistant. Given a chat history and the latest user question,
    turn it into a standalone question that can be understood without the previous messages.
    Do NOT answer the question. If the user asks about CHOTU’s identity (e.g. “Who are you?”),
    do not prepend or append any context—return the original question exactly."""
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are CHOTU, a helpful AI assistant. Use the following context to answer the user's question.
     Only include information from “Context: {context}” if it directly helps answer the question.
     If the question doesn’t require that context—such as asking about your identity—answer without referencing it."""),
    ("system", "Context: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ───────────────────────── Complete Implementation ──────────────────────────
# In your langchain_utils.py


from langchain_core.runnables import RunnablePassthrough

def get_rag_chain(model_name: str):
    """RAG chain using LangChain's chain composition"""
    
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=0
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Function to format context
    def format_context(inputs):
        docs = inputs["source_documents"]
        context = "\n\n".join([doc.page_content for doc in docs])
        return {**inputs, "context": context}
    
    # Function to generate answer
    def generate_answer(inputs):
        combined_context = (
        "You are CHOTU, a very helpful AI assistant. "
        "CHOTU’s purpose is to assist and answer questions accurately.\n\n"
        + inputs['context']
    )
        prompt = f"""you are CHOTU a very helpful AI assistant ,you are given a context and a question , you can also use the chat history to answer the question.
        reply with the answer only, do not include any other information.

Context: {combined_context}

Question: {inputs['input']}

Answer based on the context."""
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        return {**inputs, "answer": answer}
    
    # Fixed function to add sources
    def add_sources_to_response(inputs):
        answer = inputs["answer"]
        source_docs = inputs.get("source_documents", [])
        
        if source_docs:
            sources_info = "\n\n📚 **Sources:**\n"
            for i, doc in enumerate(source_docs, 1):
                file_id = doc.metadata.get('file_id', 'Unknown')
                source_file = doc.metadata.get('source_file', 'Unknown')
                chunk_idx = doc.metadata.get('chunk_index', 0)
                total_chunks = doc.metadata.get('total_chunks', 1)
                sources_info += f"• Document {i}: {source_file} (ID: {file_id}, Chunk {chunk_idx + 1}/{total_chunks})\n"
            
            # ✅ Return dict, maintain chain structure
            return {**inputs, "answer": answer + sources_info}
        
        return inputs
    
    # Create the complete chain
    chain = (
        RunnablePassthrough.assign(
            source_documents=lambda x: retriever.get_relevant_documents(x["input"])
        )
        | format_context
        | generate_answer  
        | add_sources_to_response
    )
    
    return chain

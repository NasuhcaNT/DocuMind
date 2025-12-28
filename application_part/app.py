import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- PAGE SETTINGS ---
st.set_page_config(page_title="DocuMind AI", page_icon="📄")
st.title("📄 DocuMind")
st.markdown("### Interactive RAG Application")

# Fetching the token from Environment Variables (Render Settings)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- YOUR ORIGINAL TRUNCATE FUNCTION ---
def truncate_documents(docs, max_chars=400):
    truncated_contents = []
    for doc in docs:
        content = doc.page_content
        truncated_contents.append(content[:max_chars])
    return "\n\n".join(truncated_contents)

# --- MODEL INITIALIZATION ---
@st.cache_resource
def load_rag_assets():
    # Embedding model stays the same as your original code
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Using Endpoint instead of Pipeline for Render compatibility
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large", 
        huggingfacehub_api_token=hf_token,
        temperature=0.1
    )
    return embeddings, llm

embeddings, llm = load_rag_assets()

# --- FILE UPLOAD COMPONENT ---
uploaded_file = st.file_uploader("Please upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Temporarily saving the uploaded file to process it
    with open("temp_storage", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Selecting the loader based on file extension
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader("temp_storage")
    else:
        loader = TextLoader("temp_storage")
    
    documents = loader.load()
    
    # Creating the Vector Database (In-Memory for efficiency)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    st.success("Knowledge base is ready!")

    # --- THE RAG CHAIN (Your LCEL Logic) ---
    # English prompt for better model performance
    template = """Use the provided context to answer the user's question. 
    If the answer is not in the context, say that you don't know.

    Context: {context}
    
    Question: {question}
    
    Answer (Reply in the language of the question):"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # We keep your exact chain logic here
    rag_chain = (
        {
            "context": vectorstore.as_retriever(k=1) | RunnableLambda(truncate_documents),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- CHAT INPUT ---
    user_input = st.text_input("Ask a question about your document:")
    if user_input:
        with st.spinner("Processing..."):
            response = rag_chain.invoke(user_input)
            st.markdown("#### Result:")
            st.info(response)
else:
    st.warning("Awaiting document upload...")
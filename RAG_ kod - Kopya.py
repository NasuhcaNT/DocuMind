"""
Title: SQuAD Dataset RAG Prototype
Description: A Retrieval-Augmented Generation implementation using LangChain and HuggingFace.
"""

#!pip install -q -U langchain langchain-community langchain-huggingface sentence-transformers chromadb datasets transformers

from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load the Data
print("Loading dataset...")
dataset = load_dataset("squad", split="train[:200]")
documents = list(set(dataset['context']))

# 2. Prepare the Embedding Model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create the Vector Database
print("Creating vector database (This may take a while)...")
langchain_docs = [Document(page_content=doc) for doc in documents]
vectorstore = Chroma.from_documents(documents=langchain_docs, embedding=embeddings)

print("Process completed! Ready for querying.")

"""
Testing: Run the cell below to see if the system can search through the documents:
"""

query = "What is the importance of university research?"
docs = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(docs):
    print(f"\n--- Nearest Paragraph {i+1} ---")
    print(doc.page_content[:300] + "...")

import os

# Set the Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_XXXXX_YOUR_TOKEN"

print("Hugging Face API token set as environment variable.")

# 1. Initialize a transformers pipeline locally
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400
)

# 2. Wrap the transformers pipeline with HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Define a function to truncate the document content
def truncate_documents(docs: list[Document], max_chars: int = 350) -> str:
    """Truncates the content of a list of documents and joins them into a single string."""
    truncated_contents = []
    for doc in docs:
        content = doc.page_content
        truncated_contents.append(content[:max_chars])
    return "\n\n".join(truncated_contents)

print("HuggingFacePipeline LLM initialized and truncate_documents function defined.")

# 3. Create a "Question-Answer" Template (Prompt)
template = """Context: {context}\nQuestion: {question}\nAnswer:"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Set up the Chain (Modern Method: LCEL)
# This chain: Takes the question -> searches in the vector store -> places it in the prompt -> asks the LLM -> returns as text
rag_chain = (
    {
        "context": vectorstore.as_retriever(k=1) | RunnableLambda(truncate_documents),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Run!
question_input = "When was the First Year of Studies program established?" # Test with a question directly answerable by the SQuAD data
response = rag_chain.invoke(question_input)

print("\n--- Model Response ---")
print(response)
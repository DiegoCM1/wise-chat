from dotenv import load_dotenv
load_dotenv()

import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel

# REFACTORED: We now import Chroma from LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# REFACTORED: We need the Document object to add data
from langchain.docstore.document import Document


# --- Pydantic Models (No Change) ---
class IngestData(BaseModel):
    text: str
    source: str

# ... (rest of the models are the same)
class QueryData(BaseModel):
    q: str

class IngestResponse(BaseModel):
    chunks: int
    message: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# --- App Initialization ---
app = FastAPI()

# --- LangChain & Vector Store Setup ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

# REFACTORED: Initialize LangChain's Chroma vector store
# This will create a local folder named 'chroma_db' to store the vectors
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)


# --- API Routes ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/ingest_text", response_model=IngestResponse)
def ingest_text(data: IngestData):
    chunks = [data.text[i:i+500] for i in range(0, len(data.text), 450)]
    
    # REFACTORED: Convert text chunks into LangChain Document objects
    documents = [
        Document(page_content=chunk, metadata={"source": data.source}) 
        for chunk in chunks
    ]
    
    # REFACTORED: Use the vector store's 'add_documents' method
    vectorstore.add_documents(documents)
    
    return {"chunks": len(documents), "message": f"Text from source '{data.source}' ingested."}


@app.post("/query", response_model=QueryResponse)
def query(data: QueryData):
    # REFACTORED: Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    template = """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(data.q)
    
    # REFACTORED: We need to get the source documents from the retriever
    retrieved_docs = retriever.invoke(data.q)
    sources = [doc.metadata['source'] for doc in retrieved_docs]

    return {"answer": answer, "sources": list(set(sources))}
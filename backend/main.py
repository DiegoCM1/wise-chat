from dotenv import load_dotenv

load_dotenv()
import os

print(f"--- Loaded OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')} ---")

import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document


# --- Pydantic Models to validate types of info ---
class IngestData(BaseModel):
    text: str
    source: str


class QueryData(BaseModel):
    q: str


class IngestResponse(BaseModel):
    chunks: int
    message: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


app = FastAPI()

# --- Set up of LLM, Embedding model and local vector DB ---
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)  # Initializes the embedding local model
llm = ChatOpenAI(  # Calls the API of the LLM
    model_name="x-ai/grok-4-fast:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)

vectorstore = Chroma(  # Creates a local folder named 'chroma_db' to store the vectors
    persist_directory="./chroma_db", embedding_function=embeddings
)


@app.get("/")  # Checks if the API is running
def read_root():
    return {"status": "API is running"}


# --- INGEST: Takes data and converts it into chunks/LangChain Document objects, saving them into the local Vector DB ---
@app.post("/ingest_text", response_model=IngestResponse)
def ingest_text(data: IngestData):
    chunks = [data.text[i : i + 500] for i in range(0, len(data.text), 450)]

    # Convert text chunks into LangChain Document objects
    documents = [
        Document(page_content=chunk, metadata={"source": data.source})
        for chunk in chunks
    ]

    # Use the vector store's 'add_documents' method
    vectorstore.add_documents(documents)

    return {
        "chunks": len(documents),
        "message": f"Text from source '{data.source}' ingested.",
    }


# --- QUERY: Receives the input from the user ---
@app.post("/query", response_model=QueryResponse)
def query(data: QueryData):
    # --- ADD THIS FOR DEBUGGING ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(data.q)
    print("--- DEBUG: RETRIEVED DOCS ---")
    print(retrieved_docs)
    print("-----------------------------")
    # --- END DEBUG ---

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Prompt is defined
    template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based on the context, just say that you don't know. 
    Be concise and stick to the facts.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # RAG Chain is defined
    rag_chain = (  # Creates the chain of process, passing the output of the first process as the input of the next
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(data.q)

    retrieved_docs = retriever.invoke(data.q)
    sources = [doc.metadata["source"] for doc in retrieved_docs]

    # Answer is returned
    return {"answer": answer, "sources": list(set(sources))}

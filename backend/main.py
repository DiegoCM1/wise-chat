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
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document


# --- Pydantic Models to validate types of info ---
class IngestData(BaseModel):
    text: str
    source: str


class QueryData(BaseModel):
    q: str
    k: int = 2



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


# --- QUERY: Inputs the questions, a response as output ---
@app.post("/query", response_model=QueryResponse)
def query(data: QueryData):
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": data.k})

    # Define the prompt template
    template = """
    Behave as a QA expert.
    Base your answers on the provided context.
    If you don't have enough information to answer, say "I don't have information to answer that."
    
    Context: {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # This is the chain that takes the context and question and generates an answer
    rag_chain_from_docs = (
        prompt
        | llm
        | StrOutputParser()
    )

    # This is the full chain that retrieves documents, passes them through, and gets the answer
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # Invoke the final chain. The result is a dictionary.
    result = rag_chain_with_source.invoke(data.q)
    
    # Extract the answer and sources from the single result dictionary
    answer = result["answer"]
    retrieved_docs = result["context"]
    sources = [doc.metadata["source"] for doc in retrieved_docs]

    # Add logging for observability
    print("--- LOG ---")
    print(f"Query: {data.q}")
    print(f"Retrieved {len(retrieved_docs)} documents with k={data.k}")
    print(f"Answer: {answer}")
    print("-----------")

    return {"answer": answer, "sources": list(set(sources))}

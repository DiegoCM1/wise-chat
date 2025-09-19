from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import uuid

# --- Pydantic Models for Request/Response ---
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

# --- App Initialization ---
app = FastAPI()

# --- In-Memory Database (ChromaDB) ---
# This is simple for the demo. No Docker or separate DB needed.
client = chromadb.Client()
collection = client.get_or_create_collection(name="interview_collection")

# --- API Routes ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/ingest_text", response_model=IngestResponse)
def ingest_text(data: IngestData):
    # Simple text splitting (replace with LangChain later)
    chunks = [data.text[i:i+500] for i in range(0, len(data.text), 450)] # 500 chunk, 50 overlap
    
    # Store in ChromaDB
    collection.add(
        documents=chunks,
        metadatas=[{"source": data.source} for _ in chunks],
        ids=[str(uuid.uuid4()) for _ in chunks]
    )
    
    return {"chunks": len(chunks), "message": f"Text from source '{data.source}' ingested."}


@app.post("/query", response_model=QueryResponse)
def query(data: QueryData):
    # This is a stub for now. We will implement the RAG logic in Block 2.
    print(f"Received query: {data.q}")
    # --- STUBBED RESPONSE ---
    return {"answer": "This is a placeholder answer.", "sources": ["source1.txt", "source2.pdf"]}
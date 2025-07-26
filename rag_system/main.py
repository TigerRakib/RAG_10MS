import fitz  # PyMuPDF
import json
import os
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from langchain_community.chat_models import ChatOllama 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PDF_PATH = "D:\\project_10ms\data\HSC26-Bangla1st-Paper.pdf"
CHUNKS_FILE = "paragraph_chunks.json"
QDRANT_METADATA_FILE = "qdrant_metadata.json"
COLLECTION_NAME = "my_collection"
MAX_CHARS = 1000

app = FastAPI()

# Initialize global resources
embedding_model = SentenceTransformer("BAAI/bge-m3")
qdrant_client = QdrantClient(host="localhost", port=6333)
llm = ChatOllama(model="phi") 

def is_bangla(text: str) -> bool:
    return any('\u0980' <= ch <= '\u09FF' for ch in text)

def clean_bangla_text(text: str) -> str:
    
    cleaned = re.sub(r"[^\u0980-\u09FF\s\u09E6-\u09EF.,;:!?\"'()-]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()

def split_paragraph(text: str, max_chars: int):
    if not max_chars or len(text) <= max_chars:
        return [text.strip()]
    return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]

def extract_layout_paragraphs(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_id = 1
    para_threshold = 10

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        paragraph_lines = []
        last_y = None

        for b in blocks:
            text = b[4].strip()
            if not text or not is_bangla(text):
                continue

            y0 = b[1]
            lines = text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if last_y is not None and abs(y0 - last_y) > para_threshold:
                    full_para = " ".join(paragraph_lines).strip()
                    full_para = clean_bangla_text(full_para)
                    for para in split_paragraph(full_para, MAX_CHARS):
                        chunks.append({
                            "chunk_id": chunk_id,
                            "type": "paragraph",
                            "page": page_num,
                            "content": para
                        })
                        chunk_id += 1
                    paragraph_lines = []

                paragraph_lines.append(line)
                last_y = y0

        # Add last paragraph of the page
        if paragraph_lines:
            full_para = " ".join(paragraph_lines).strip()
            full_para = clean_bangla_text(full_para)
            for para in split_paragraph(full_para, MAX_CHARS):
                chunks.append({
                    "chunk_id": chunk_id,
                    "type": "paragraph",
                    "page": page_num,
                    "content": para
                })
                chunk_id += 1

    return chunks

def create_embeddings_and_qdrant(chunks, model, client, collection_name):
    texts = [f"Represent this document for retrieval: {chunk['content']}" for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    # Recreate collection fresh (deletes existing)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    points = []
    for chunk, vector in zip(chunks, embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload=chunk
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    return client, embeddings, chunks

def retrieve_top_k_chunks(query, model, client, collection_name, k=5):
    query_text = f"Represent this document for retrieval: {query}"
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=k
    )
    return [hit.payload for hit in search_result]

def generate_answer(context_chunks, user_query):
    context = "\n\n".join([chunk["content"] for chunk in context_chunks])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful Bengali assistant. Based on the context below, answer the question **briefly and exactly** in Bengali.

Context:
{context}

Question:
{question}

Answer only with the exact answer in Bengali:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        return chain.invoke({"context": context, "question": user_query})
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Pydantic model for query request
class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    # Extract paragraphs and create embeddings if needed!!
    if not os.path.exists(CHUNKS_FILE):
        print("Extracting paragraphs from PDF...")
        chunks = extract_layout_paragraphs(PDF_PATH)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print("Creating embeddings and uploading to Qdrant...")
        create_embeddings_and_qdrant(chunks, embedding_model, qdrant_client, COLLECTION_NAME)
        with open(QDRANT_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    else:
        print("Chunks file found, skipping extraction.")

@app.get("/")
def read_root():
    return {"message": "Bangla PDF QA with FastAPI and Qdrant"}

@app.post("/ask/")
def query_qa(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_chunks = retrieve_top_k_chunks(question, embedding_model, qdrant_client, COLLECTION_NAME)
    answer = generate_answer(top_chunks, question)
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": top_chunks
    }

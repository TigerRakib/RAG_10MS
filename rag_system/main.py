
import fitz  # PyMuPDF
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
MAX_CHARS = 1000

def is_bangla(text):
    return any('\u0980' <= ch <= '\u09FF' for ch in text)

def split_paragraph(text, max_chars):
    if not max_chars or len(text) <= max_chars:
        return [text.strip()]
    return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]

def extract_layout_paragraphs(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_id = 1
    para_threshold = 10  # vertical distance in pixels

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # sort by Y, then X

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

        # flush remaining
        if paragraph_lines:
            full_para = " ".join(paragraph_lines).strip()
            for para in split_paragraph(full_para, MAX_CHARS):
                chunks.append({
                    "chunk_id": chunk_id,
                    "type": "paragraph",
                    "page": page_num,
                    "content": para
                })
                chunk_id += 1

    return chunks


def create_embeddings_and_qdrant(chunks, model_name, client, collection_name):

    # Prepare texts for embedding
    texts = [f"Represent this document for retrieval: {chunk['content']}" for chunk in chunks]

    # Create normalized embeddings for cosine similarity
    embeddings = model_name.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    dim = embeddings.shape[1]

    # Create (or recreate) collection with cosine similarity
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    # Prepare points with unique IDs, vectors, and payloads (metadata)
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload=chunk 
        )
        points.append(point)

    # Upload points to Qdrant collection
    client.upsert(collection_name=collection_name, points=points)

    return client, embeddings, chunks

def retrieve_top_k_chunks(query, model, qdrant_client, collection_name, k=5):
    # Embed query
    query_text = f"Represent this document for retrieval: {query}"
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]

    # Search Qdrant
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=k
    )

    return [hit.payload for hit in search_result]
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOllama(model="llama2:7b-chat")
def generate_answer(context_chunks, user_query):
    context = "\n\n".join([chunk["content"] for chunk in context_chunks])
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.invoke({"context": context, "question": user_query})


if __name__ == "__main__":
    pdf_file = "data/HSC26-Bangla1st-Paper.pdf"
    chunks_output_file = "paragraph_chunks.json"
    metadata_file = "qdrant_metadata.json" 
    collection_name = "my_collection"       
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(host="localhost", port=6333)

    print("Extracting paragraphs...")
    chunks = extract_layout_paragraphs(pdf_file)

    print(f"Saving chunks to {chunks_output_file}")
    with open(chunks_output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Creating embeddings and uploading to Qdrant...")
    with open(chunks_output_file, encoding="utf-8") as f:
        chunks = json.load(f)

    # create_embeddings_and_qdrant returns Qdrant client, embeddings, metadata
    client, embeddings, metadata = create_embeddings_and_qdrant(chunks,embedding_model,client, collection_name=collection_name)

    # file to save with Qdrant
    print(f"Saving metadata to {metadata_file} (optional)")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    top_chunks = retrieve_top_k_chunks(query, embedding_model, client, collection_name)
    answer = generate_answer(top_chunks, query)
    print("Answer:", answer)
    print("Done.")
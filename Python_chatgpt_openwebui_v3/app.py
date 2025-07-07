import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import HTTPException, Request
from uuid import uuid4
import time, json

from vector_store import create_index, load_index
from chain import ask_question
from parse_message import parse_message


app = FastAPI()

print("CWD:", os.getcwd())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing/dev; in prod, set allowed origins!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./data")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PRACTICE_FILE_DIR = Path("practice_file")

@app.on_event("startup")
def index_practice_files():
    """Index all PDFs in the practice_file directory at startup."""
    if not PRACTICE_FILE_DIR.exists():
        print(f"❌ {PRACTICE_FILE_DIR.resolve()} does not exist.")
        return
    for pdf in PRACTICE_FILE_DIR.glob("*.pdf"):
        file_id = pdf.stem  # Or use uuid4() for unique index per run
        print(f"→ Indexing: {pdf} as {file_id}")
        try:
            create_index(str(pdf), file_id)
            print(f"✓ Indexed: {pdf.name}")
        except Exception as e:
            print(f"✗ Failed to index {pdf.name}: {e}")

# ── /v1/models for WebUI’s model dropdown ─────────────────────────────────────
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "my-awesome-model",  # appears in WebUI
                "object": "model",
                "created": 0,
                "owned_by": "local-pdf-rag",
            }
        ],
    }

# ── OpenAI‑compatible schema ───────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False  # If you want streaming, you'll need to implement it



@app.post("/api/v1/files")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = UPLOAD_DIR / file.filename
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    file_id = str(uuid4())
    create_index(str(pdf_path), file_id)

    return {"message": "Indexed successfully", "file_id": file_id}

@app.post("/api/v1/ask")
async def ask_endpoint(question: str, file_id:str):
    try:
        print("Error at ask_endpoint---file_id repr:", repr(file_id))
        db = load_index(file_id)
        docs = db.similarity_search(question)
        answer = ask_question(docs, question)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": answer}



@app.post("/v1/chat/completions")
async def chat_endpoint(payload: ChatRequest, request: Request = None):
    """
    OpenAI-compatible chat endpoint for OpenWebUI.
    Handles both streaming and non-streaming modes.
    """
    # Find the most recent user message
    try:
        user_msg = next(m.content for m in reversed(payload.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(status_code=400, detail="No user message provided")

    # Parse file_id and question from the user message
    file_id, question = parse_message(user_msg)
    if not file_id:
        raise HTTPException(status_code=400, detail=(
            "Missing file_id. Ask your question like: "
            '"for file_id=\\"YOURID\\" What is in this PDF?"'
        ))

    # Retrieve relevant docs and answer
    try:
        db = load_index(file_id)
        docs = db.similarity_search(question)
        answer = ask_question(docs, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {e}")

    # STREAMING MODE
    if getattr(payload, "stream", False):
        def event_stream():
            data = {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": payload.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            # End of stream signal
            data["choices"][0]["delta"] = {}
            data["choices"][0]["finish_reason"] = "stop"
            yield f"data: {json.dumps(data)}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # NON-STREAMING MODE
    return {
        "id": f"chatcmpl-{uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,  # Optional: If you can, compute actual usage!
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }



from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from loader import load_and_split


INDEX_DIR = Path("./faiss_index")
EMBEDDER = OpenAIEmbeddings(openai_api_key="")

def create_index(pdf_path, file_id=None):
    chunks = load_and_split(pdf_path)
    db = FAISS.from_documents(chunks, EMBEDDER)
    index_dir = INDEX_DIR / (file_id or "default")
    index_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_dir))
    return db

PROJECT_ROOT = Path(__file__).parent.resolve()

def load_index(file_id=None):
    print("from load_index file_id repr:", repr(file_id))
    if not file_id:
        raise ValueError("file_id must be provided to load the correct FAISS index directory.")

    index_dir = PROJECT_ROOT / "faiss_index" / file_id
    print("Looking for:", index_dir.resolve())

    if not index_dir.exists():
        raise FileNotFoundError(
            f"NO FAISS directory found for file_id='{file_id}'. Please upload and index a PDF first.")

    db = FAISS.load_local(str(index_dir), EMBEDDER, allow_dangerous_deserialization=True)
    return db
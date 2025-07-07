from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        raise ValueError("No pages loaded from PDF")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = splitter.split_documents(docs)
    return chunks
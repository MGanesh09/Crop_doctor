import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

KNOWLEDGE_BASE_FOLDER = str(BASE_DIR / "KnowledgeBase")
FAISS_INDEX_PATH = os.path.join(KNOWLEDGE_BASE_FOLDER, "faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "models/embedding-001")


def build_index():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY is not set in environment or .env file.")
        print("Please add GOOGLE_API_KEY to your .env file to generate embeddings.")
        return

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError as e:
        print(f"❌ Missing dependency: {e}. Please install requirements via 'pip install -r requirements.txt'")
        return

    if not os.path.exists(KNOWLEDGE_BASE_FOLDER):
        print(f"❌ Error: KnowledgeBase directory not found at {KNOWLEDGE_BASE_FOLDER}")
        return

    docs = []
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"ℹ️ No PDF documents found in {KNOWLEDGE_BASE_FOLDER}.")
        print("Pre-built FAISS index files (index.faiss, index.pkl) are already provided in the repository.")
        return

    for file in pdf_files:
        path = os.path.join(KNOWLEDGE_BASE_FOLDER, file)
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
            print(f"  Loaded: {file}")
        except Exception as e:
            print(f"  Warning: Failed to load {file}: {e}")

    if not docs:
        print("❌ No readable content extracted from PDFs.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks created.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL, google_api_key=google_api_key
    )
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vs.save_local(FAISS_INDEX_PATH)
    print("🎉 FAISS index built and saved successfully to:", FAISS_INDEX_PATH)


if __name__ == "__main__":
    build_index()

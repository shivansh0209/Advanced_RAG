import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.hybrid_text_splitters import parent_document_retriever_setup
from src.data_loaders  import load_documents
from src.generate_answers import generate_answer
import os

load_dotenv()

# ---- Config ----
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
DATA_PATH = "data"
COLLECTION_NAME = "research_papers"
PERSIST_DIR = "chroma_db"

def main():
    # 1. Load documents
    docs = load_documents(path=DATA_PATH)
    print(f"Loaded {len(docs)} documents")

    # 2. Build hybrid parent-child retriever
    retriever = parent_document_retriever_setup(
        docs,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    query = input("\nAsk a question: ")
    results = retriever.invoke(query)
    answer = generate_answer(query, results)
    print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
import warnings
from dotenv import load_dotenv
from langsmith import traceable
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.hybrid_text_splitters import parent_document_retriever_setup
from src.generate_answers import generate_answer
from src.utils import get_hyde_answer_and_refined_prompt

import os

load_dotenv()

# ---- Config ----
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
DATA_PATH = "data"
COLLECTION_NAME = "acts"
PERSIST_DIR = "chroma_db"

@traceable(name="ADVANCED RAG AGENT")
def main():
    retriever = parent_document_retriever_setup(
        docs=None,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    while True:
        query = input("\nAsk a question: ")
        hyde_answer, refined_query = get_hyde_answer_and_refined_prompt(query)

        context = retriever.invoke(refined_query)
        answer = generate_answer(query, context)

        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
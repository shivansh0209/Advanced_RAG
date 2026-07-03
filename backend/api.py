from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from src.hybrid_text_splitters import parent_document_retriever_setup
from src.generate_answers import generate_answer
from src.utils import get_hyde_answer_and_refined_prompt

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = parent_document_retriever_setup(
    collection_name="acts",
    persist_directory="chroma_db"
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    _, refined_query = get_hyde_answer_and_refined_prompt(q.question)
    context = retriever.invoke(refined_query)
    answer = generate_answer(q.question, context)
    acts = list(set(doc.metadata.get("act_name", "") for doc in context))
    return { "answer": answer, "acts_cited": acts }
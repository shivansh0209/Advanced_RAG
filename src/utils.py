import os
import cohere
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
    

def get_hyde_answer(query, model_name="mistral"):
    model = ChatOllama(model=model_name)

    prompt = PromptTemplate.from_template(
        "Please write a comprehensive, technical paragraph answering the following question. "
        "Do not include introductory phrasing or pleasantries, just provide the direct answer.\n\n"
        "Question: {question}"
    )

    parser = StrOutputParser()

    chain = prompt | model | parser
    
    return chain.invoke({"question": query})

def rerank_with_cohere(query, documents, top_k=6):
    cohere_client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
    rerank_response = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=top_k
        )

    return [documents[result.index] for result in rerank_response.results]
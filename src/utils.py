import os
import cohere
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import re

ACT_NAMES = {
    "data/CRPC1973.pdf": "Code of Criminal Procedure, 1973",
    "data/aa2005.pdf": "Right to Information Act, 2005",
    "data/A2013-18.pdf": "Companies Act, 2013",
    "data/A1955-25Eng.pdf": "Hindu Marriage Act, 1955",
    "data/A2000-21 (1).pdf": "Information Technology Act, 2000",
    "data/A187209.pdf": "Indian Contract Act, 1872",
    "data/AA1860-21.pdf": "Indian Penal Code, 1860",
    "data/eng201935.pdf": "Consumer Protection Act, 2019"
}


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
    
@traceable(name="get_hyde_answer")
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

@traceable(name="rerank_with_cohere")
def rerank_with_cohere(query, documents, top_k=6):
    if(not os.environ.get("COHERE_API_KEY")):
        return documents
    else:
        print("Reranking with Cohere API...")
    cohere_client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
    rerank_response = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=top_k
        )

    return [documents[result.index] for result in rerank_response.results]


def split_by_legal_section(docs):
    from collections import defaultdict
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source")].append(doc)

    section_docs = []
    section_pattern = re.compile(r'\n(?=\d{1,4}[A-Z]?\.\s)')

    for source, pages in grouped.items():
        full_text = "\n".join(p.page_content for p in pages)
        parts = section_pattern.split(full_text)

        act_name = ACT_NAMES.get(source, source)  # fallback to filename if not mapped

        for part in parts:
            part = part.strip()
            if not part:
                continue
            match = re.match(r'^(\d{1,4}[A-Z]?)\.\s', part)
            section_num = match.group(1) if match else None

            # prepend act name + section into the actual text
            enriched_content = f"{act_name}, Section {section_num}: {part}" if section_num else f"{act_name}: {part}"

            section_docs.append(
                Document(
                    page_content=enriched_content,
                    metadata={"source": source, "section": section_num, "act_name": act_name}
                )
            )
    return section_docs
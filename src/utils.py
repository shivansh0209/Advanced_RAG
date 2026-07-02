import os
import cohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import re

ACT_NAMES = {
    "data/CRPC1973.pdf": "code of criminal procedure, 1973",
    "data/aa2005.pdf": "right to information act, 2005",
    "data/A2013-18.pdf": "indian contract act, 2013",
    "data/A1955-25Eng.pdf": "hindu marriage act, 1955",
    "data/it_act_2000.pdf": "information technology act, 2000",
    "data/AA1860-21.pdf": "indian penal code, 1860",
    "data/cpa.pdf": "consumer protection act, 2019"
}


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
    
@traceable(name="Hyde and Refine Prompt")
def get_hyde_answer_and_refined_prompt(query, model_name="mistral"):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    prompt_for_hyde = PromptTemplate.from_template(
    "You are an Indian legal expert. Write a technical paragraph answering the question using precise statutory language.\n"
    "Rules:\n"
    "- Do NOT guess section numbers — describe provisions in plain statutory language if unsure\n"
    "- Use exact terminology found in Indian statutes\n"
    "- If multiple acts are relevant, address each separately\n\n"
    "Question: {question}"
)

    prompt_for_refinement = PromptTemplate.from_template(
        "You are a search query optimizer for Indian statute text.\n"
        "Rewrite the query to maximize keyword and semantic retrieval across ALL relevant acts.\n"
        "Rules:\n"
        "- Expand abbreviations (CrPC → Code of Criminal Procedure 1973, IPC → Indian Penal Code 1860)\n"
        "- Add legal synonyms (arrest → custody, detained, apprehended)\n"
        "- Add statutory terminology likely appearing verbatim in the acts\n"
        "- If query spans multiple acts, include relevant terms from EACH act separately\n"
        "- DO NOT invent section numbers\n"
        "- Output only the rewritten query, nothing else\n\n"
        "Query: {query}"
    )

    parser = StrOutputParser()

    chain_for_hyde = prompt_for_hyde | model | parser
    chain_for_refinement = prompt_for_refinement | model | parser

    # hyde_answer = chain_for_hyde.invoke({"question": query})
    refined_query = chain_for_refinement.invoke({"query": query})

    return None, refined_query


@traceable(name="Cohere Reranking")
def rerank_with_cohere(query, documents, top_k=6):
    if(not os.environ.get("COHERE_API_KEY")):
        return documents
    cohere_client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
    rerank_response = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=top_k
        )

    return [documents[result.index] for result in rerank_response.results]


def split_by_legal_section(docs, max_parent_chars=2000):
    from collections import defaultdict
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source")].append(doc)

    section_docs = []
    section_pattern = re.compile(r'\n(?=\d{1,4}[A-Z]?\.\s)')

    # Matches subsections like (1), (a), (i), (ii) — common in legal text
    subsection_pattern = re.compile(r'\n(?=\([\w]+\)\s)')

    for source, pages in grouped.items():
        full_text = "\n".join(p.page_content for p in pages)
        parts = section_pattern.split(full_text)
        act_name = ACT_NAMES.get(source, source)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            match = re.match(r'^(\d{1,4}[A-Z]?)\.\s', part)
            section_num = match.group(1) if match else None
            prefix = f"{act_name}, Section {section_num}" if section_num else act_name

            enriched_content = f"{prefix}: {part}"

            # ── If section fits within limit, store as-is ──────────────────
            if len(enriched_content) <= max_parent_chars:
                section_docs.append(Document(
                    page_content=enriched_content,
                    metadata={"source": source, "section": section_num, "act_name": act_name}
                ))
                continue

            # ── Section too large: try splitting on subsections first ───────
            subsections = subsection_pattern.split(part)

            if len(subsections) > 1:
                # Accumulate subsections into chunks under the char limit
                current_chunk = ""
                sub_index = 0

                for sub in subsections:
                    sub = sub.strip()
                    if not sub:
                        continue

                    candidate = f"{prefix} (part {sub_index + 1}): {current_chunk}\n{sub}".strip()

                    if len(candidate) <= max_parent_chars:
                        current_chunk = f"{current_chunk}\n{sub}".strip()
                    else:
                        # Save what we have, start a new chunk
                        if current_chunk:
                            section_docs.append(Document(
                                page_content=f"{prefix} (part {sub_index + 1}): {current_chunk}",
                                metadata={"source": source, "section": section_num,
                                          "act_name": act_name, "part": sub_index + 1}
                            ))
                            sub_index += 1
                        current_chunk = sub

                # Save the last remaining chunk
                if current_chunk:
                    section_docs.append(Document(
                        page_content=f"{prefix} (part {sub_index + 1}): {current_chunk}",
                        metadata={"source": source, "section": section_num,
                                  "act_name": act_name, "part": sub_index + 1}
                    ))

            else:
                # No subsections found — hard split by character limit with overlap
                overlap = 340
                step = max_parent_chars - overlap
                for i, start in enumerate(range(0, len(enriched_content), step)):
                    chunk_text = enriched_content[start:start + max_parent_chars]
                    section_docs.append(Document(
                        page_content=chunk_text,
                        metadata={"source": source, "section": section_num,
                                  "act_name": act_name, "part": i + 1}
                    ))

    print(f"[split_by_legal_section] Total parent chunks created: {len(section_docs)}")
    return section_docs
import os
import cohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import re
from collections import defaultdict

ACT_NAMES = {
    "data/crpc_1973.md":    "code of criminal procedure, 1973",
    "data/rti_act_2005.md": "right to information act, 2005",
    "data/A1955-25.md":     "hindu marriage act, 1955",
    "data/it_act_2000.md":  "information technology act, 2000",
    "data/cpa_2019.md":          "consumer protection act, 2019",
    "data/contract.md":     "indian contract act, 1872",
    "data/ipc_act.md":      "indian penal code, 1860",
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
        "You are a search query optimizer for Indian statute text used in a hybrid BM25 + vector retrieval system.\n\n"
        "TASK: Rewrite the query to maximize retrieval of the most relevant statutory chunks.\n\n"
        "STEP 1 — CLASSIFY the query into one or more of these types:\n"
        "  [DEFINITIONAL]   — what is X, meaning of X, define X, what does X mean\n"
        "  [PROCEDURAL]     — how, steps, process, when, produce, arrest, file, appeal\n"
        "  [PENALTY]        — punishment, offence, fine, imprisonment, liable, convicted\n"
        "  [RIGHTS]         — can I claim, entitled to, right to, relief available, remedy\n"
        "  [DISTINCTION]    — difference between, distinguish, X vs Y, when does X become Y\n"
        "  [EXCEPTION]      — exception to, when does X not apply, proviso, notwithstanding\n"
        "  [OVERLAP]        — two or more acts mentioned, where do acts intersect or conflict\n"
        "  [HYPOTHETICAL]   — named parties, scenario with multiple legal issues across acts\n"
        "  [OTHER]          — none of the above, general query, broad question, multiple acts\n\n"
        "STEP 2 — EXPAND abbreviations always:\n"
        "  CrPC → Code of Criminal Procedure 1973\n"
        "  IPC  → Indian Penal Code 1860\n"
        "  IT Act → Information Technology Act 2000\n"
        "  HMA  → Hindu Marriage Act 1955\n"
        "  RTI  → Right to Information Act 2005\n"
        "  CPA  → Consumer Protection Act 2019\n"
        "  ICA  → Indian Contract Act 1872\n\n"
        "STEP 3 — APPLY type-specific expansion rules:\n\n"
        "  [DEFINITIONAL]:\n"
        "    - Include verbatim anchors: 'means', 'shall mean', 'defined as', 'includes', 'definition'\n"
        "    - CrPC defined terms (cognizable, bailable, warrant, summons, charge): add 'Section 2' and 'Schedule I'\n"
        "    - IPC defined terms: add 'Chapter II' (general definitions section of IPC)\n"
        "    - IT Act defined terms: add 'Section 2' (definitions section of IT Act)\n"
        "    - HMA defined terms: add 'Section 3' (definitions section of HMA)\n"
        "    - CPA defined terms: add 'Section 2' (definitions section of CPA)\n"
        "    - ICA defined terms: add 'Section 2' (definitions section of ICA)\n\n"
        "  [PROCEDURAL]:\n"
        "    - Add: 'procedure', 'manner', 'shall', 'officer', 'magistrate', 'within'\n"
        "    - Arrest cluster: 'without warrant', 'custody', 'produce before magistrate', 'twenty-four hours', 'grounds of arrest'\n"
        "    - Bail cluster: 'bail', 'bond', 'bailable', 'non-bailable', 'release'\n"
        "    - Investigation cluster: 'investigation', 'inquiry', 'First Information Report', 'cognizance'\n"
        "    - Trial cluster: 'trial', 'charge', 'summons', 'warrant', 'sessions court'\n"
        "    - Appeal cluster: 'appeal', 'revision', 'appellate authority', 'Commission'\n"
        "    - RTI cluster: 'public authority', 'Public Information Officer', 'thirty days', 'Central Information Commission'\n\n"
        "  [PENALTY]:\n"
        "    - Add: 'punishable', 'imprisonment', 'fine', 'liable', 'conviction', 'shall be punished'\n"
        "    - IPC homicide cluster: 'culpable homicide', 'murder', 'intention', 'knowledge', 'bodily injury', 'likely to cause death'\n"
        "    - IPC fraud/forgery cluster: 'cheating', 'dishonestly', 'forgery', 'makes false document', 'fraudulently'\n"
        "    - IPC marriage offences cluster: 'bigamy', 'void marriage', 'sapinda', 'prohibited degrees'\n"
        "    - IPC modesty cluster: 'outraging modesty', 'assault', 'criminal force', 'voyeurism', 'stalking'\n"
        "    - IT Act cyber offences cluster: 'computer resource', 'electronic record', 'data', 'unauthorised access'\n"
        "    - IT Act privacy cluster: 'violation of privacy', 'publishes', 'transmits', 'intimate image'\n"
        "    - IT Act identity cluster: 'identity theft', 'electronic signature', 'impersonation', 'password'\n"
        "    - CPA penalty cluster: 'complaint', 'District Commission', 'opposite party', 'unfair trade practice'\n\n"
        "  [RIGHTS]:\n"
        "    - Add: 'entitled', 'right', 'may claim', 'relief', 'remedy', 'shall be liable to pay'\n"
        "    - CPA rights cluster: 'deficiency in service', 'District Commission', 'complainant', 'Section 39', 'compensation'\n"
        "    - CPA product liability cluster: 'product manufacturer', 'product seller', 'harm', 'defect', 'Chapter VI'\n"
        "    - RTI rights cluster: 'information', 'public authority', 'applicant', 'deemed refusal'\n"
        "    - ICA rights cluster: 'breach of contract', 'damages', 'specific performance', 'void agreement'\n\n"
        "  [DISTINCTION]:\n"
        "    - Include both terms being distinguished in full statutory language\n"
        "    - Add the section numbers of both provisions if well-known (e.g. Section 299 and Section 300 IPC)\n"
        "    - Add: 'difference', 'distinguished', 'amounts to', 'does not amount to'\n\n"
        "  [EXCEPTION]:\n"
        "    - Add: 'exception', 'proviso', 'notwithstanding', 'shall not apply', 'nothing in this section'\n"
        "    - Include the parent provision being excepted from in full statutory name\n\n"
        "  [OVERLAP]:\n"
        "    - Name each act in full\n"
        "    - Add 1-2 verbatim terms per act separately — do NOT blend terms across acts\n"
        "    - Add: 'in addition to', 'not in derogation of', 'concurrent', 'inconsistency'\n"
        "    - CPA + IT Act overlap anchor: 'Section 100 Consumer Protection Act', 'Section 43A Information Technology Act'\n\n"
        "  [HYPOTHETICAL]:\n"
        "    - Identify each distinct legal issue in the scenario separately\n"
        "    - Expand each issue using its relevant act's verbatim terms\n"
        "    - Do NOT blend terms — keep each act's expansion self-contained\n"
        "    - Treat as CROSS-ACT only for acts explicitly implicated by the facts\n\n"
        "STEP 4 — HARD CONSTRAINTS:\n"
        "  - Do NOT invent section numbers\n"
        "  - Do NOT add terms unless confident they appear verbatim in Indian statutes\n"
        "  - Single-act queries: stay within that act, do NOT import terms from other acts\n"
        "  - Keep rewritten query under max(60 words, original query length + 10)\n"
        "  - Output ONLY the rewritten query — no labels, no classification, no explanation\n\n"
        "Query: {query}"
    )
    parser = StrOutputParser()

    chain_for_hyde = prompt_for_hyde | model | parser
    chain_for_refinement = prompt_for_refinement | model | parser

    refined_query = chain_for_refinement.invoke({"query": query})

    return None, refined_query


@traceable(name="Cohere Reranking")
def rerank_with_cohere(query, documents, top_k=6):
    if not os.environ.get("COHERE_API_KEY"):
        return documents
    cohere_client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
    rerank_response = cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[doc.page_content for doc in documents],
        top_n=top_k
    )
    return [documents[result.index] for result in rerank_response.results]


# ── Compiled patterns ────────────────────────────────────────────────────────
_SECTION_START = re.compile(
    r'(?:(?:<sup>\d+</sup>)?\s*\[?\s*)?'
    r'\*\*'
    r'\[?'
    r'(\d{1,4}[A-Z]?)'
    r'\.'
    r'(?!\s*</sup>)',
    re.MULTILINE,
)

_FOOTNOTE_LINE = re.compile(
    r'^[ \t]*(?:'
    r'<sup>\d+\.?</sup>\.?\s+'           # <sup>1.</sup> or <sup>1</sup>.
    r'|'
    r'\d{1,3}\.\s+'                       # bare  1.  2.  12.
    r'(?:Subs|Ins|Omit|Rep|Added|The\s|Cl\.|Proviso|See|vide|In\s|For\s|This\s|Certain|Words?|Clause)'
    r')'
    r'.*$',
    re.MULTILINE,
)

_PAGE_NUMBER = re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE)
_HTML_TABLE = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
_PREAMBLE_END = re.compile(r'\n# THE ', re.IGNORECASE)
_TOC_LINE = re.compile(
    r'^[ \t]*\d{1,4}[A-Z]?\.\s+(?!\*\*)[A-Z*\[][^—\n]*$',
    re.MULTILINE,
)

_SUBSECTION = re.compile(r'\n(?=\([\w]+\)\s)')


def _clean_markdown(text: str) -> str:
    text = _HTML_TABLE.sub('', text)
    text = _FOOTNOTE_LINE.sub('', text)
    text = _TOC_LINE.sub('', text)
    text = _PAGE_NUMBER.sub('', text)
    text = re.sub(r'<sup>\d+</sup>', '', text)
    text = re.sub(r'\\\*\\\*\\\*', '', text)
    text = re.sub(r'(?m)^[ \t]*\*[ \t]*\*[ \t]*\*.*$', '', text)
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_by_legal_section(docs, max_parent_chars: int = 4500) -> list[Document]:
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source")].append(doc)

    section_docs = []

    for source, pages in grouped.items():
        full_text = "\n".join(p.page_content for p in pages)
        act_name = ACT_NAMES.get(source, source)

        preamble_matches = list(_PREAMBLE_END.finditer(full_text))
        if preamble_matches:
            best_idx, best_count = 0, -1
            for i, m in enumerate(preamble_matches):
                next_pos = preamble_matches[i + 1].start() if i + 1 < len(preamble_matches) else len(full_text)
                count = len(_SECTION_START.findall(full_text[m.start():next_pos]))
                if count > best_count:
                    best_count, best_idx = count, i
            full_text = full_text[preamble_matches[best_idx].start():]

        first_section = _SECTION_START.search(full_text)
        if first_section and first_section.start() > len(full_text.split('\n')[0]) + 5:
            heading_end = full_text.find('\n') + 1
            full_text = full_text[:heading_end] + full_text[first_section.start():]

        # ── Find all section boundaries ───────────────────────────────────────
        boundaries = [(m.start(), m.group(1)) for m in _SECTION_START.finditer(full_text)]

        if not boundaries:
            cleaned = _clean_markdown(full_text)
            if cleaned:
                section_docs.append(Document(
                    page_content=f"{act_name}: {cleaned}",
                    metadata={"source": source, "section": None, "act_name": act_name}
                ))
            continue

        # ── Slice out each section and clean it ───────────────────────────────
        for i, (start, section_num) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            cleaned = _clean_markdown(full_text[start:end])
            if not cleaned:
                continue

            prefix = f"{act_name}, Section {section_num}"
            enriched = f"{prefix}: {cleaned}"

            # Fits within limit — store as-is
            if len(enriched) <= max_parent_chars:
                section_docs.append(Document(
                    page_content=enriched,
                    metadata={"source": source, "section": section_num, "act_name": act_name}
                ))
                continue

            # Too large — try splitting on subsection markers first
            subsections = _SUBSECTION.split(cleaned)

            if len(subsections) > 1:
                current_chunk = ""
                sub_index = 0
                MIN_CHUNK_CHARS = 300   # never emit a parent chunk smaller than this
                for sub in subsections:
                    sub = sub.strip()
                    if not sub:
                        continue
                    candidate = f"{prefix} (part {sub_index + 1}): {current_chunk}\n{sub}".strip()
                    if len(candidate) <= max_parent_chars:
                        current_chunk = f"{current_chunk}\n{sub}".strip()
                    else:
                        if current_chunk and len(current_chunk) >= MIN_CHUNK_CHARS:
                            section_docs.append(Document(
                                page_content=f"{prefix} (part {sub_index + 1}): {current_chunk}",
                                metadata={"source": source, "section": section_num,
                                          "act_name": act_name, "part": sub_index + 1}
                            ))
                            sub_index += 1
                            current_chunk = sub
                        else:
                            # Too small to emit alone — absorb into next chunk
                            current_chunk = f"{current_chunk}\n{sub}".strip()
                if current_chunk:
                    section_docs.append(Document(
                        page_content=f"{prefix} (part {sub_index + 1}): {current_chunk}",
                        metadata={"source": source, "section": section_num,
                                  "act_name": act_name, "part": sub_index + 1}
                    ))
            else:
                # No subsections — hard split with overlap
                overlap = 450
                step = max_parent_chars - overlap
                for j, start_char in enumerate(range(0, len(enriched), step)):
                    section_docs.append(Document(
                        page_content=enriched[start_char: start_char + max_parent_chars],
                        metadata={"source": source, "section": section_num,
                                  "act_name": act_name, "part": j + 1}
                    ))

    print(f"[split_by_legal_section] Total parent chunks created: {len(section_docs)}")
    return section_docs
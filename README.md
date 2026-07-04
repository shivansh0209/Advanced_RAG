# ⚖️ LexAI — Advanced RAG System for Indian Legal Statutes

> Hybrid BM25 + Vector Retrieval · Legal-Aware Chunking · Cohere Reranking  
> LangSmith Observability · HyDE Query Refinement · LLM-as-a-Judge Evaluation

---

## 📌 Overview

LexAI is a production-grade Retrieval-Augmented Generation (RAG) system built specifically for Indian legal statutes. It allows users to query across seven major Indian Acts simultaneously and receive precise, citation-backed answers grounded in the actual legislative text.

Every architectural decision was made to maximise retrieval accuracy on a highly factual, domain-specific legal corpus — moving well beyond the basic split → embed → retrieve baseline.

---

## 📚 Covered Statutes

| Abbreviation | Act |
|---|---|
| IPC 1860 | Indian Penal Code, 1860 |
| CrPC 1973 | Code of Criminal Procedure, 1973 |
| HMA 1955 | Hindu Marriage Act, 1955 |
| RTI Act 2005 | Right to Information Act, 2005 |
| IT Act 2000 | Information Technology Act, 2000 |
| ICA 1872 | Indian Contract Act, 1872 |
| CPA 2019 | Consumer Protection Act, 2019 |

All source PDFs are sourced from [indiacode.nic.in](https://indiacode.nic.in) — the official Government of India legislative repository.

---

## 🏗️ System Architecture

```
USER QUERY
    │
    ▼
QUERY REWRITING (Groq / LLaMA 3.3 70B)
Abbreviations resolved · Synonyms added · Multi-act terms expanded
    │
    ├─────────────────────┬─────────────────────┐
    ▼                     ▼                     │
VECTOR RETRIEVAL      BM25 RETRIEVAL           │
(Chroma, top-36)     (Keyword, top-18)         │
    │                     │                     │
    └──────────┬──────────┘                     │
               ▼                                │
    RECIPROCAL RANK FUSION                      │
    Parent deduplication                        │
               │                                │
               ▼                                │
    CROSS-ACT DIVERSITY (Pass 1)                │
    1 slot guaranteed per act                   │
               │                                │
               ▼                                │
    COHERE RERANKING (rerank-v3.5)              │
    Cross-encoder, top-15 → top-10              │
               │                                │
               ▼                                │
    CROSS-ACT DIVERSITY (Pass 2)                │
    Max 3 chunks per act                        │
               │                                │
               ▼                                │
    ANSWER GENERATION (Groq / LLaMA 3.3 70B)   │
    Grounded · Citation-backed                  │
               │                                │
               ▼                                │
    LANGSMITH TRACE ◄───────────────────────────┘
    Every step logged
```

---

## 🔬 What Makes This an Advanced RAG System

### 1. Legal-Domain-Aware Parent-Child Chunking

Instead of blindly splitting by character count, a custom `split_by_legal_section()` function parses PDFs on actual section boundaries (e.g. `5.`, `12A.`), preserving each section as a single semantic parent unit.

- Sections > 3,000 chars are split on subsection markers like `(1)`, `(a)`, `(i)`
- Every chunk is prefixed with its Act name and section number: `"Indian Penal Code, 1860, Section 302: ..."`
- Child chunks (400 chars, 60-char overlap) are embedded; matched children return their full parent to the LLM

This is the **small-to-big retrieval pattern** — small chunks for precise matching, large parents for full context.

### 2. Hybrid BM25 + Vector Retrieval with Reciprocal Rank Fusion

| Engine | Strength |
|--------|----------|
| BM25 | Exact keyword match — critical for legal jargon and section numbers |
| Chroma Vectorstore | Semantic similarity — handles paraphrased queries |

Both engines run independently. Results are merged with **Reciprocal Rank Fusion**:

```
RRF score = Σ 1 / (k + rank),  where k = 60
```

RRF rewards documents ranked highly in *both* engines simultaneously.

### 3. Cross-Act Diversity Enforcement

Naive retrieval clusters results from one act. The retriever enforces diversity in two passes:

- **Pass 1:** Guarantee at least one chunk per relevant act before reranking
- **Pass 2:** Cap at 3 chunks per act in the final top-10

### 4. HyDE Query Refinement (Without Hallucination Risk)

Standard HyDE (generate a fake answer, embed it) is dangerous for factual legal databases — a hallucinated section number sends the retriever in the wrong direction.

LexAI uses **HyDE Query**, not HyDE Answer:
- The rewritten query goes to the **vectorstore** (richer semantic signal)
- The original query goes to **BM25** (preserving keyword exactness)

### 5. Cohere Reranking

After hybrid retrieval produces up to 15 parent documents, Cohere `rerank-v3.5` (a cross-encoder) reads the query and each document *together* for a much more accurate relevance score than bi-encoder similarity alone.

---

## 🛠️ Technology Stack

| Category | Tool | Role |
|---|---|---|
| LLM Inference | Groq API (LLaMA 3.3 70B) | Query rewriting, answer generation |
| Embeddings | sentence-transformers (local) | all-MiniLM-L6-v2, no API cost |
| Vector Store | ChromaDB (persisted) | Dense vector index for child chunks |
| Keyword Search | BM25Retriever (LangChain) | Exact keyword matching |
| Reranking | Cohere rerank-v3.5 | Cross-encoder precision uplift |
| Orchestration | LangChain | Chains, retrievers, prompt templates |
| Observability | LangSmith | Full trace of every retrieval step |
| PDF Loading | PyPDF + pdfplumber | Text extraction |
| Docstore | LocalFileStore (LangChain) | Persistent parent document store |

---

## 📊 Evaluation Results

Evaluated using **LLM-as-a-Judge** — Claude (Anthropic API) scores answers on factual correctness against manually verified ground truth drawn directly from the statute text.

| System Version | Score | Key Change |
|---|---|---|
| Baseline (naive chunking, vector-only) | 64% | Initial run |
| + Legal section chunking + parent-child | ~72% | Domain-aware chunking |
| + BM25 hybrid + RRF | ~79% | Hybrid retrieval |
| **+ Cohere reranking + HyDE query + diversity** | **85%** | **Full system ✓** |

### Evaluation Methodology

- 10 questions spanning all 7 acts, designed for section-specific factual retrieval
- Corpus-wide questions deliberately excluded (those require Graph RAG)
- Ground truth manually verified against actual PDF text
- Scored on 4 dimensions: factual correctness, section citation accuracy, completeness, hallucination
- Failures categorised by root cause: data quality, chunking boundaries, hallucination

---

## 🔍 Data Quality Assurance

All source PDFs are validated through `data_check.py` before indexing:

| Check | What It Catches |
|---|---|
| File size | Truncated or empty files |
| Page count | Partial/amendment-only acts |
| Font embedding analysis | Non-embedded fonts → garbled extraction |
| Text extraction yield | Scanned image PDFs |
| Garble ratio | > 5% replacement characters → rejected |
| Content keyword verification | Wrong act entirely (e.g. Companies Act instead of Contract Act) |

This script caught PDFs with garble ratios of 6.9% that would have silently polluted the index.

---

## ⚙️ Hyperparameter Rationale

| Parameter | Value | Reason |
|---|---|---|
| Child chunk size | 400 chars | Sweet spot for single sub-clause capture |
| Child overlap | 60 chars | Prevents boundary split loss |
| Parent max size | 3,000 chars | Fits complete section within LLM attention window |
| vector_fetch_k | 36 | Wide semantic net |
| bm25_fetch_k | 18 | Already keyword-precise, smaller set sufficient |
| reranking_k | 15 | ~50% compression before final top-10 |
| top_k | 10 | ~10–15 pages of statute text in LLM context |
| MAX_PER_ACT | 3 | Prevents single-act dominance |
| RRF k | 60 | Original paper value; 2× spread between rank-1 and rank-60 |

---

## 🔭 LangSmith Observability

Every component is decorated with `@traceable`. Traced components:

- `ADVANCED_RAG_AGENT` — top-level trace per query
- `HybridRetriever.invoke` — full retrieval call
- `BM25Retriever` — keyword retrieval sub-trace
- `vectorstore_retrieval` — dense retrieval sub-trace
- `Hyde_and_Refine_Prompt` — query rewriting trace
- `Cohere_Reranking` — reranker I/O
- `generate_answer` — LLM call with context and response

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
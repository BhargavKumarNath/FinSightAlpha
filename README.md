![LangChain](https://img.shields.io/badge/AI--Framework-LangChain-1C3C3C?style=flat-square&logo=langchain)
![LangGraph](https://img.shields.io/badge/Agents-LangGraph-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![RAG Architecture](https://img.shields.io/badge/Architecture-RAG-blueviolet)
![Financial Analysis](https://img.shields.io/badge/Focus-Financial_Insights-gold)
![GitHub top language](https://img.shields.io/github/languages/top/bhargavkumarnath/finsightalpha)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


## 🚀 Live Demo
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://finsightalpha.streamlit.app/)

# FinSight-Alpha: Comprehensive System Architecture & Design Analysis

> **A production-grade Agentic RAG system for institutional-quality financial document intelligence.**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture & Workflow](#2-architecture--workflow)
3. [Data & Knowledge Base](#3-data--knowledge-base)
4. [Agentic Capabilities](#4-agentic-capabilities)
5. [Key Features & Innovations](#5-key-features--innovations)
6. [What Elevates This System to Enterprise Grade](#6-what-elevates-this-system-to-enterprise-grade)
7. [Dataset Flexibility — Using Any Dataset](#7-dataset-flexibility--using-any-dataset)
8. [Limitations & Improvements](#8-limitations--improvements)

---



## 1. System Overview

### What It Does

**FinSight-Alpha** is an Agentic Retrieval-Augmented Generation (RAG) system purpose built for institutional grade financial document analysis. It ingests, indexes, and reasons over financial filings. These include SEC EDGAR submissions (10-K, 10-Q, 8-K) and supporting PDF, HTML, and text documents. It exposes a conversational interface through which users can pose complex, multi-hop financial queries.

The system's defining characteristic is that it does **not** simply embed a query and return a chunk. It deploys a full reasoning *agent* that plans, decomposes, retrieves, synthesizes, and self-corrects before emitting a final, cited response.

### Core Objectives

| Objective | Implementation |
|---|---|
| Grounded financial reasoning | Strictly context-bound answers with `[Doc X: source]` citations |
| Multi-hop query resolution | LangGraph agent with iterative Plan→Rewrite→Retrieve loops |
| Production reliability | Graceful degradation, error isolation, budget-aware routing |
| High retrieval precision | Hybrid dense+sparse search with cross-encoder reranking |
| Token efficiency | Three-tier budget management, semantic caching, model routing |
| Scalability | Multi-collection Qdrant, async ingestion, incremental indexing |

### Use Cases

- **Risk analysis**: "What supply chain risks does NVIDIA identify in its most recent 10-K?"
- **Comparative financials**: "Compare NVIDIA data center revenue across FY23 and FY24."
- **Multi-hop entity reasoning**: "Identify the CEO of the company that acquired Figma in 2022, then list the primary language used in the open-source framework that company originally created."
- **Hallucination testing**: "What does the 10-K say about NVIDIA's secret plans to acquire AMD?" → Agent correctly finds no evidence and reports it.

---
## 2. Architecture & Workflow

### High-Level System Map

```
┌─────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                              │
│   Streamlit UI (app.py / pages/)  ←→  FastAPI Server (main.py)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ POST /chat
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OPTIMIZATION LAYER                             │
│  SemanticResponseCache → (hit) → return cached answer           │
│  TokenBudgetManager   → tier (GREEN/YELLOW/RED)                 │
│  ModelRouter          → light (8B) or heavy (70B) model         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (cache miss)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AGENTIC CORE (LangGraph)                       │
│                                                                 │
│  ┌──────────┐   ┌─────────────┐   ┌───────────┐                 │
│  │ Planner  │──▶│QueryRewriter│──▶│ Retriever │                 │
│  └──────────┘   └─────────────┘   └─────┬─────┘                 │
│                                         │                       │
│  ┌───────────┐   ┌──────────┐           │                       │
│  │ Responder │◀──│Reflector │◀──────────┤                       │
│  └───────────┘   └──────────┘   ┌──────▼──────┐                 │
│        │              │         │   Reasoner  │                 │
│        ▼         (loop back)    └─────────────┘                 │
│   Final Answer   if needed                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HybridRetriever.search()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RETRIEVAL LAYER                                │
│  Dense: Qdrant (all-MiniLM-L6-v2, 384-dim cosine vectors)       │
│  Sparse: BM25Okapi (per-collection .pkl)                        │
│  Fusion: Reciprocal Rank Fusion (RRF, k=60)                     │
│  Rerank: CrossEncoder ms-marco-MiniLM-L-6-v2                    │
│  Cache:  LRU embedding cache + TTL result cache                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (index time)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INGESTION LAYER                                │
│  ParserRegistry → PDFParser / HTMLParser / SECEdgarParser /     │
│                   TextParser / JSONParser                       │
│  SemanticChunker → section/page/flat strategy + overlap         │
│  CollectionManager → Qdrant collection lifecycle                │
│  DocumentRegistry  → incremental indexing (content-hash check)  │
│  BM25 rebuild      → after each batch ingestion                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Data Flow: Query to Response

#### Step 0: API Entry (`src/main.py`)

The `POST /chat` endpoint receives a `QueryRequest`. Before touching the LLM pipeline, it checks the **SemanticResponseCache**. This cache embeds the incoming query and computes cosine similarity against all stored query embeddings. If `similarity ≥ 0.92` (configurable), the cached response, reasoning trace, and synthetic metrics (`MRR=1.0, NDCG=1.0`) are returned immediately, bypassing the entire agent graph. This is a **full-pipeline bypass** zero LLM calls.

```python
cached = response_cache.get(request.query)
if cached is not None:
    return QueryResponse(answer=..., cache_hit=True, tokens_used=0)
```

#### Step 1: Planner Node (`langgraph_agent.py::planner_node`)

A fast **llama-3.1-8b-instant** model (via Groq) receives the raw user query and the `PLANNER_PROMPT` (Senior Financial Strategist persona). It outputs a 1–3 step research plan in plain English. This node is **skipped entirely** under YELLOW and RED budget tiers to save ~800 tokens.

```
Budget tier GREEN  → planner runs, produces: "1. Find supply chain risks. 2. Identify fiscal year."
Budget tier YELLOW → plan = "Synthesize directly." (planner bypassed)
```

#### Step 2: Query Rewriter Node (`query_rewriter_node`)

The same **8B model** receives the plan and `REWRITER_PROMPT`. It decomposes the original question into atomic sub-queries optimized for vector database semantic search. Crucially:

- It uses **sequential dependency rules**: If sub-query B requires the answer to sub-query A, they are ordered correctly.
- It outputs strict `{"queries": ["q1", "q2", ...]}` JSON.
- Reflection feedback from previous loops is injected here, making the rewriter *adaptive* across iterations.

**Example decomposition** (from the test harness in the CLI):
```
Input:  "Identify the CEO of the company that acquired Figma in 2022..."
Output: ["Company that acquired Figma in 2022",
         "CEO of Adobe",
         "Open-source framework created by Adobe"]
```

#### Step 3: Retriever Node (`retriever_node`)

For each sub-query, `HybridRetriever.search()` is called. The retrieval pipeline is:

```
sub_query
   │
   ├─► [Dense]  embed query (LRU cache) → Qdrant cosine search (fetch_k=50)
   │
   └─► [Sparse] tokenize → BM25Okapi.get_scores() → top-50 by score
          │
          └─► RRF fusion (k=60) → top-20 candidates
                    │
                    └─► CrossEncoder rerank (ms-marco-MiniLM-L-6-v2)
                              │
                              └─► top_n results sorted by cross-encoder score
```

Results from all sub queries are **deduplicated** by text (exact match), sorted by score descending, and accumulated with `[Doc X: source_file]` identifiers for citation tracking. Under GREEN tier, `top_n=8`; YELLOW→5; RED→4.

#### Step 4: Reasoner Node (`reasoner_node`)

The **llama-3.3-70b-versatile** model (swapped to 8B under RED tier) receives:
- The original user query
- All retrieved chunks formatted as `[Doc X: filename] (Relevance: Y.YY)\n{text}`

The `REASONER_PROMPT` enforces strict rules: cite every claim with `[Doc X: source]`, include specific numbers when present, do not fabricate. The output is a `draft_answer`.

#### Step 5: Reflector Node (`reflector_node`)

The **8B model** (always light, as this is a classification task) evaluates the draft against truncated context (first 400 chars per chunk to save tokens). It returns structured JSON:

```json
{
  "is_grounded": true,
  "needs_more_info": false,
  "feedback": "All claims are directly supported by Doc 1 and Doc 3."
}
```

The `route_reflection` conditional edge then decides:

| Condition | Next Node |
|---|---|
| `is_grounded=true` OR `loop_count >= max_loops` | `responder` (finalize) |
| `needs_more_info=true` | `query_rewriter` (fetch more context) |
| `is_grounded=false` AND `needs_more_info=false` | `reasoner` (rewrite answer only) |

`max_loops = budget_manager.get_max_iterations() // 2` → 3 full reflect loops under GREEN.

#### Step 6: Responder Node (`responder_node`)

**Happy path**: wraps the verified `draft_answer` into an `AIMessage`.

**Error path (Graceful Degradation)**: If any upstream node set `state["error"]`, the responder detects it and invokes the heavy LLM directly with a generic financial assistant prompt, with no RAG and citations. A system note is appended indicating the bypass. This ensures the API **never returns a 500** to the user under normal operating conditions.

#### Step 7: Response Packaging & Metrics (`main.py`)

The server extracts the final answer, builds the reasoning trace (`Plan`, `Searched for:`, `Reflection` entries), computes live **MRR** and **NDCG** scores from citation overlap with the ranked retrieval list, caches the result in `SemanticResponseCache`, and returns the full `QueryResponse`.

---

### Module Interaction Map

```
main.py (FastAPI)
  │
  ├── SemanticResponseCache   ◄─── shared all-MiniLM-L6-v2 model
  ├── TokenBudgetManager      ◄─── session-scoped, thread-safe
  ├── ModelRouter             ◄─── reads budget tier
  │
  └── build_graph() (LangGraph)
        │
        ├── planner_node      → ModelRouter.get_planner_llm()  (8B)
        ├── query_rewriter_node → ModelRouter.get_planner_llm() (8B)
        ├── retriever_node    → HybridRetriever.search()
        │     │
        │     └── HybridRetriever
        │           ├── _EmbeddingCache (LRU, 256 entries)
        │           ├── _ResultCache    (TTL=300s)
        │           ├── Qdrant client   (local path)
        │           ├── BM25Okapi       (per-collection .pkl)
        │           └── CrossEncoder    (ms-marco-MiniLM-L-6-v2)
        │
        ├── reasoner_node     → ModelRouter.get_agent_llm()    (70B or 8B)
        ├── reflector_node    → ModelRouter.get_planner_llm()  (8B)
        └── responder_node    → ModelRouter.get_agent_llm()    (fallback only)
```


---

## 3. Data & Knowledge Base

### Document Types Supported

The **ParserRegistry** (`src/ingestion/parsers/registry.py`) implements the **Strategy pattern**. It acts as a central router that selects the most appropriate parser for each file using a two-pass resolution algorithm.

1. **Content-based detection** (priority parsers): Checks if a parser's `can_parse()` override matches (e.g., `SECEdgarParser` inspects the first 2000 bytes for `<SEC-DOCUMENT>`, `ACCESSION NUMBER`, or `<ix:` XBRL tags).
2. **Extension-based lookup**: Falls back to the `.ext → parser` dictionary.
3. **Fallback**: `TextParser` handles anything unrecognised.

| Parser | Formats | Key Capabilities |
|---|---|---|
| `SECEdgarParser` | `.txt` (SEC EDGAR full-submission files) | Extracts `<DOCUMENT>` blocks, skips binary/XBRL attachments (`GRAPHIC`, `ZIP`, `EX-101.*`), delegates HTML blocks to `HTMLParser`, extracts filing metadata (accession number, CIK, period of report, company name) |
| `PDFParser` | `.pdf` | Dual-engine: **pdfplumber** (primary, table extraction) → **PyMuPDF/fitz** (fallback for scanned/complex PDFs). Per-page extraction with page-number tracking, table formatting as pipe-delimited text, hyphenation repair, page-number artifact removal |
| `HTMLParser` | `.html`, `.htm` | BeautifulSoup4 extraction, XBRL tag stripping, table-to-text conversion |
| `TextParser` / `JSONParser` | `.txt`, `.md`, `.json`, `.jsonl` | Plain text normalization, JSONL chunk-format support |

Every parser produces a canonical **`ParsedDocument`**. This is the universal contract between the parsing and chunking layers. It contains:
- `content` — full clean text
- `pages: List[ParsedPage]` — per page content with page numbers
- `sections: List[DocumentSection]` — hierarchically detected sections
- `metadata: DocumentMetadata` — source path, title, author, dates, content hash, page count

### Section Detection

`BaseParser._detect_sections()` applies regex patterns to identify structural boundaries:

```
SEC items:       "Item 1A. Risk Factors"        → level 1
PART headers:    "PART II"                       → level 1
Markdown:        "## Management Discussion"      → level = # count
Separator:       "=== SECTION ==="              → level 1
```

Sections carry a `page_start` reference and a `content_type` classification (`NARRATIVE`, `TABLE`, `FINANCIAL`, `LEGAL`, `HEADER`, `FOOTNOTE`).

### Chunking Strategy (`src/ingestion/chunking.py`)

`SemanticChunker` selects one of three strategies per document:

| Condition | Strategy | Logic |
|---|---|---|
| `has_sections` AND `≥ 3 sections` | **Section-based** | Chunks respect section boundaries; large sections split with overlap; hierarchical `section_path` maintained |
| `has_pages` (multi-page) | **Page-based** | Per-page chunking with page number tracking |
| Flat document | **Flat** | Simple paragraph-boundary splitting |

**Splitting details** (`_split_with_overlap`):
- Max chunk: **1500 chars** (configurable)
- Overlap: **200 chars** carried from the tail of the previous chunk to the head of the next
- Split priority: paragraph break (`\n\n`) → sentence boundary (`[.!?]\s`) → last whitespace
- Minimum chunk size: **80 chars**

**Noise filtering** (`_is_noise`) rejects:
- Chunks shorter than 80 chars
- HTML-heavy text (>30% tag tokens)
- XBRL schema definitions (`parentTag`, `schemaRef`, `DefinitionEquity`)
- Pure JSON objects
- Low alphabetic ratio (<30% alpha characters in strings >50 chars)

Every retained chunk carries a `ChunkMetadata` object with 12 fields:

```python
@dataclass
class ChunkMetadata:
    source_path: str       # Full file path
    source_name: str       # Filename (for display)
    chunk_index: int       # Position within document
    total_chunks: int      # Total chunks from this document
    section_header: str    # Nearest section heading
    section_path: List[str]# Hierarchical path ["PART II", "Item 7"]
    page_number: int       # PDF page number
    page_range: str        # e.g., "12-14"
    content_type: str      # "narrative", "financial", etc.
    content_hash: str      # SHA-256[:12] for deduplication
    document_title: str    # From parsed metadata
    document_hash: str     # Document-level SHA-256[:16]
```

This rich metadata flows all the way to the Qdrant payload and enables future **metadata filtering** at retrieval time.

### Storage & Indexing

#### Qdrant Vector Store (`data/qdrant_db/`)

Local on-disk Qdrant database using the `qdrant-client` library. Each document chunk is stored as a `PointStruct`:
```python
PointStruct(
    id=<sequential integer>,
    vector=<384-dim float32 list>,   # all-MiniLM-L6-v2 embedding
    payload={"text": "...", "metadata": {...}}
)
```

**Multi-collection support**: `CollectionManager` creates isolated Qdrant collections per dataset (e.g., `sec_filings`, `nvidia_10k_fy2026`). Each collection has its own vector index with `Distance.COSINE`.

#### BM25 Sparse Index (`data/bm25_index.pkl`, `data/bm25_{collection}.pkl`)

After each batch ingestion, the pipeline scrolls **all points** from the target Qdrant collection and rebuilds a `BM25Okapi` index from scratch. The index is serialized as a `(bm25, corpus_metadata)`. This dual ensures the sparse index stays synchronized with the dense index.

The `HybridRetriever` supports per-collection BM25 with a lazy-loading in-memory cache: `_bm25_cache: Dict[str, tuple]`.

#### Document Registry (`data/.registry_{collection}.json`)

A JSON-backed persistent registry tracks every indexed document:
```json
{
  "abc123def456": {
    "doc_id": "abc123def456",
    "content_hash": "a1b2c3d4e5f6g7h8",
    "source_path": "data/raw/sec-edgar-filings/...",
    "chunk_count": 47,
    "indexed_at": 1745750000.0,
    "collection": "sec_filings",
    "point_ids": [0, 1, 2, ..., 46]
  }
}
```

**Incremental indexing**: before parsing a document, the pipeline calls `registry.is_indexed(content_hash, collection)`. A SHA-256 hash of the full document content is computed, if already present, the file is skipped entirely. The `force=True` flag overrides this behaviour for re-indexing.

---

## 4. Agentic Capabilities

### Agent Structure (LangGraph StateGraph)

FinSight-Alpha uses **LangGraph** to define a directed, stateful graph.

```python
class AgentState(TypedDict):
    messages:       Annotated[Sequence[BaseMessage], add_messages]  # append-only
    original_query: str
    plan:           str
    sub_queries:    List[str]
    context_chunks: Annotated[List[Dict], operator.add]  # append across loops
    draft_answer:   str
    reflection:     str
    is_grounded:    bool
    loop_count:     int
    latencies:      Annotated[Dict[str, float], add_latencies]  # custom reducer: sum
    error:          str
```

Crucially, `context_chunks` uses `operator.add` as its reducer, meaning chunks from *all* retrieval passes across reflection loops are **accumulated** — the agent never discards previously retrieved context when looping back for more.

### Graph Topology

```
START
  │
  ▼
[planner]
  │
  ▼
[query_rewriter] ◄───────────────┐
  │                              │
  ▼                              │ (needs_retrieval)
[retriever]                      │
  │                              │
  ▼                              │
[reasoner] ◄───────────────┐     │
  │                        │     │ (hallucinated)
  ▼                        │     │
[reflector]                │     │
  │                        │     │
  ├── is_grounded OR loop_count >= max ───────► [responder] ───► END
  │
  ├── needs_retrieval ────────────────────────► [query_rewriter]
  │
  └── hallucinated ──────────────────────────► [reasoner]


ERROR PATH (global override):
Any node ──► state["error"] ──► [responder] (LLM fallback, no RAG, no citations)
```

### Tool Usage & Reasoning Loops

FinSight-Alpha does **not** use traditional LangChain tool-calling. Instead, retrieval is baked directly into the `retriever_node`. This was a deliberate architectural decision after earlier iterations suffered from LLM tool calling hallucinations, where the model would appear to invoke a tool without actually executing it.

The current design enforces tool execution via the graph topology: the LLM has **no choice** but to follow the Plan→Rewrite→Retrieve path. The agent's "tools" are the graph edges themselves.

### Decision-Making Logic

The `route_reflection` function is the heart of agentic decision making:

```python
def route_reflection(state: AgentState):
    # Error bypass → graceful degradation
    if state.get("error"):
        return "responder"

    is_grounded = state.get("is_grounded", False)
    loop_count  = state.get("loop_count", 0)
    max_loops   = budget_manager.get_max_iterations() // 2  # budget-aware

    # Budget or quality gate
    if is_grounded or loop_count >= max_loops:
        return "responder"

    # Reflector said "need more docs"
    if "needs_retrieval" in reflection:
        return "query_rewriter"  # re-decompose + re-retrieve

    # Reflector said "answer is hallucinated"
    return "reasoner"  # re-reason with existing context
```

This implements a **self-correcting reasoning loop** with three exit conditions:
1. Quality satisfied (`is_grounded=True`)
2. Budget exhausted (`loop_count >= max_loops`)
3. Error at any node (immediate bypass to responder)

### Memory Handling

| Memory Type | Implementation | Scope |
|---|---|---|
| **In-context (short-term)** | `AgentState.context_chunks` accumulator | Per query, across reflection loops |
| **In-context messages** | `AgentState.messages` with `add_messages` reducer | Conversation turn |
| **Semantic response cache (mid-term)** | `SemanticResponseCache` — embedding similarity LRU, TTL=1hr | Cross-session, in-process |
| **Retrieval result cache (short-term)** | `_ResultCache` in `HybridRetriever` — TTL=300s | Per-process, per query hash |
| **Embedding cache (session)** | `_EmbeddingCache` in `HybridRetriever` — LRU 256 | Per-process |
| **Document registry (persistent)** | JSON files per collection on disk | Permanent across restarts |
| **BM25 index (persistent)** | Pickle files per collection on disk | Permanent across restarts |
| **Qdrant vectors (persistent)** | Local Qdrant DB on disk | Permanent across restarts |

The system has **no conversational memory** across turns (no chat history injected into subsequent queries). Each query is treated as an independent invocation. 

---

## 5. Key Features & Innovations

### 5.1 Three-Layer Caching Architecture

FinSight-Alpha implements caching at **three distinct levels**, each targeting a different bottleneck:

#### Layer 1: Semantic Response Cache (`src/optimization/response_cache.py`)
The outermost cache. Keyed by query *meaning*, not exact text. Uses `all-MiniLM-L6-v2` to embed incoming queries and computes batch cosine similarity against all stored embeddings in a single `np.dot()` call. A match at `similarity ≥ 0.92` triggers a **full pipeline bypass**, zero LLM calls and zero retrieval.

- Capacity: 500 entries (LRU eviction)
- TTL: 1 hour (expired entries pruned on access)
- Token tracking: each entry records `token_cost`, allowing `total_tokens_saved` stat
- Thread-safe: `threading.RLock()` protects all mutations

#### Layer 2: Retrieval Result Cache (`HybridRetriever._ResultCache`)
Caches the final `top_n` results for a `(query, top_n, fetch_k)` triple. Keyed by MD5 of this composite key. TTL=5 minutes. Avoids repeated dense+sparse+rerank computation for the same query within a session.

#### Layer 3: Embedding LRU Cache (`HybridRetriever._EmbeddingCache`)
Caches raw query embeddings (numpy arrays) using an `OrderedDict` for O(1) LRU. Capacity=256. Avoids re-encoding the same query string when it appears in multiple agent loops or repeated searches.

**Combined effect**: A warm-cache query costs ~50ms (embedding lookup + dict check) vs. ~3–8s for a cold query (embed + Qdrant + BM25 + CrossEncoder + LLM × 4 nodes).

---

### 5.2 Token Budget Management (`src/optimization/token_budget.py`)

The `TokenBudgetManager` is a **production rate-limit survival system** designed specifically for Groq's free-tier constraints. It tracks cumulative token usage per session and automatically degrades the pipeline as limits approach.

```
Budget Usage       Tier      Behaviors
─────────────────────────────────────────────────────────────────
0% – 60%          GREEN     Full pipeline (70B model, 8 chunks, 6 iterations, planner on)
60% – 85%         YELLOW    Planner skipped, 5 chunks, 3 iterations, 70B model
85% – 100%        RED       Planner skipped, 4 chunks, 2 iterations, 8B model for ALL tasks
```

Each LLM call records its estimated token usage via `estimate_tokens(text)` (4 chars ≈ 1 token heuristic). The `_call_log` stores the last 10 calls for debugging. `can_afford(n)` lets any component pre-check before making an expensive call.

All parameters are overridable via `FINSIGHT_OPT_*` environment variables — zero code changes for tuning.

---

### 5.3 Tiered Model Router (`src/optimization/model_router.py`)

The `ModelRouter` lazily initialises two Groq ChatGroq instances and routes tasks to the cheapest capable model:

| Task | GREEN/YELLOW | RED |
|---|---|---|
| Planner, QueryRewriter, Reflector | `llama-3.1-8b-instant` | `llama-3.1-8b-instant` |
| Reasoner, Responder (fallback) | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` |

Lazy initialization means the heavy 70B model is **only instantiated if actually needed**, preventing unnecessary API connections at startup. Models are singletons within the router, reused across all agent invocations.

---

### 5.4 Dynamic Context Windowing (`src/optimization/context_window.py`)

Rather than injecting all retrieved chunks at full length (a common naive RAG mistake), `DynamicContextWindow` applies a **secondary relevance filter**:

1. Encodes the query and all candidate chunks in a single batch call
2. Computes cosine similarity (dot product on normalized vectors)
3. Filters chunks below `relevance_floor=0.15`
4. Takes top-K by similarity score
5. Truncates each chunk to `max_chunk_chars=1500` at **sentence boundaries** (not mid-word)
6. Enforces a hard `max_total_tokens=6000` cap — stops adding chunks if the budget would be exceeded, attempting to fit a partial chunk if >100 chars remain

This eliminates low-signal padding from the context window, reducing noise injected into the reasoner.

---

### 5.5 Hybrid Retrieval with RRF + Cross-Encoder Reranking

The retrieval pipeline combines three complementary signals:

**Dense retrieval** (Qdrant): Captures semantic similarity. `all-MiniLM-L6-v2` encodes queries into 384-dimensional cosine-space vectors. Excellent for paraphrased or conceptual queries.

**Sparse retrieval** (BM25Okapi): Captures lexical overlap. Essential for precise financial figures, ticker symbols, and named entities like "ACCESSION NUMBER 0001045810-24-000015".

**Reciprocal Rank Fusion** (RRF, k=60): Merges both ranked lists without requiring score normalization. Formula: `score(d) = Σ 1/(k + rank(d))`. Produces a fused top-20 candidates list.

**Cross-Encoder Reranking** (`ms-marco-MiniLM-L-6-v2`): Jointly encodes each `(query, chunk)` pair for precise relevance scoring. Unlike bi-encoders, the cross-encoder sees both texts simultaneously, dramatically improving precision at the cost of latency (justified since it runs on a small candidate set of 20).

The shared `_shared_embedding_model` and `_shared_cross_encoder` class variables ensure these ~80MB models are **loaded once** and reused across all `HybridRetriever` instances in the process.

---

### 5.6 Async Ingestion Pipeline (`src/ingestion/pipeline.py`)

The `IngestionPipeline` exposes both `async` and sync-wrapper APIs:

```python
# Async (preferred)
result = await pipeline.ingest_file("nvidia_10k.pdf", collection="sec_filings")
result = await pipeline.ingest_directory("data/raw/", collection="sec_filings")

# Sync convenience wrappers
result = pipeline.ingest_file_sync("nvidia_10k.pdf")
```

CPU-bound operations (parsing, embedding) are offloaded to a `ThreadPoolExecutor(max_workers=4)` via `asyncio.get_event_loop().run_in_executor()`, keeping the event loop unblocked. Embedding happens in configurable batches of 100 chunks per Qdrant upsert call, reducing round-trip overhead.

---

### 5.7 RAGAS Evaluation Framework (`src/evaluation/ragas_evaluator.py`)

An automated quality gate using two RAGAS metrics:

| Metric | What it measures | Implementation detail |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | LLM-as-judge (Groq 70B via `LangchainLLMWrapper`) |
| **Answer Relevancy** | Does the answer address the question? | Embedding-based (HuggingFace MiniLM) |

**Engineering optimizations within the evaluator:**
- Pre-computes all test query embeddings in a **single batch** (`QueryBatcher.batch_embed()`) before iterating — saves N individual encode calls
- Resets `TokenBudgetManager` between each test query to prevent inter-query budget bleed
- `GroqSafeWrapper` intercepts `_generate()` to force `n=1` (Groq API doesn't support `n>1`), maintaining RAGAS compatibility
- Quality gate: warns if average faithfulness < 0.70

---

### 5.8 Production-Grade FastAPI Server

`src/main.py` exposes a clean REST API:

| Endpoint | Method | Purpose |
|---|---|---|
| `/chat` | POST | Main query endpoint with cache + agent |
| `/health` | GET | Model routing info + budget stats |
| `/cache/stats` | GET | Hit rate, size, tokens saved |
| `/cache/clear` | POST | Flush semantic cache |
| `/budget/reset` | POST | Reset token budget for new session |

The server computes **live MRR and NDCG** from citation overlap:
- Parses `[Doc X:]` tags from the final answer to identify cited document IDs
- Walks the ranked retrieval list to find the rank of the first cited doc (MRR)
- Sums discounted gains for all cited docs (DCG) and normalizes by IDCG (NDCG)

---

## 6. What Elevates This System to Enterprise Grade

### 6.1 Engineering Quality

**Modularity**: Every layer is independently replaceable. Adding a new document format requires only subclassing `BaseParser` and calling `registry.register()` — zero changes to chunking, indexing, or agent code. The `ParsedDocument` contract enforces this boundary.

**Separation of concerns** is strict across 7 distinct modules:
```
ingestion/    → Parse, Chunk
retrieval/    → Index, Search, Fuse, Rerank
agents/       → Plan, Reason, Reflect
optimization/ → Budget, Route, Cache, Window, Batch
evaluation/   → Measure
ui/           → Present
main.py       → Orchestrate (API)
```

**Defensive programming throughout**:
- Every LangGraph node wraps its body in `try/except` and sets `state["error"]` on failure
- The responder node checks `state["error"]` and activates graceful degradation
- Parser registry has a fallback parser
- PDF parser has a fallback engine (pdfplumber → PyMuPDF)
- JSON parsing in the agent has inline fallbacks with warnings
- BM25 loading has path fallback (collection-specific → default)

**Type safety**: `TypedDict` for agent state, `@dataclass` for all data transfer objects, `Pydantic BaseModel` for API request/response schemas. No raw dicts crossing module boundaries without schemas.

**Thread safety**: `TokenBudgetManager` uses `threading.Lock()`. `SemanticResponseCache` uses `threading.RLock()`. Both are safe for concurrent FastAPI request handling.

---

### 6.2 System Design Principles

| Principle | Where Applied |
|---|---|
| **Single Responsibility** | Each LangGraph node does exactly one thing, each parser handles one format |
| **Open/Closed** | Parser system is open for extension (add new parsers), closed for modification (existing parsers untouched) |
| **Dependency Inversion** | `IngestionPipeline` accepts injected `CollectionManager`, `ParserRegistry`, `SemanticChunker` which is fully testable with mocks |
| **DRY (Don't Repeat Yourself)** | `OptimizationConfig` is a single source of truth for all tuning parameters, shared embedding model singleton pattern |
| **Fail Fast, Fail Safe** | Nodes fail immediately on exception, system falls back to direct LLM at responder level |
| **Observability First** | Every node records `latencies`, budget manager logs every call; cache reports hits/misses/tokens saved |

---

### 6.3 Comparison with Industry-Standard RAG Systems

| Capability | Naive RAG | LlamaIndex/LangChain RAG | FinSight-Alpha |
|---|---|---|---|
| Retrieval | Single-query dense only | Dense or sparse | Hybrid dense+sparse+RRF+CrossEncoder |
| Query handling | Single pass | Single pass | Multi-hop decomposition with planning |
| Self-correction | None | None | Reflector → routing loop |
| Token management | None | None | 3-tier budget + adaptive model routing |
| Caching | None | Basic | 3-layer (semantic, result, embedding) |
| Error handling | Crash | Exception bubbles | Graceful degradation with fallback LLM |
| Ingestion | Manual | Basic loaders | Async + incremental + multi-format + multi-collection |
| Evaluation | None | Optional | Built-in RAGAS (faithfulness + relevancy) |
| Observability | Logs | Logs | Live dashboard with MRR/NDCG/latency waterfall |

---

## 7. Dataset Flexibility — Using Any Dataset

> **FinSight-Alpha is architecturally dataset-agnostic.** The SEC EDGAR focus is a *default*. The entire ingestion, retrieval, and agent stack beneath is completely generic.

### 7.1 What Works Out of the Box

The `ParserRegistry.default()` already supports five formats. Any files in these formats can be ingested into a named collection immediately:

| Format | Extensions | Example datasets |
|---|---|---|
| PDF | `.pdf` | Annual reports, earnings call transcripts, research papers, central bank publications |
| HTML | `.html`, `.htm` | News articles, investor relations pages, scraped web content |
| Plain text / Markdown | `.txt`, `.md` | Regulatory guidance, news feeds, internal memos |
| JSON / JSONL | `.json`, `.jsonl` | Pre-chunked LangChain datasets (`page_content` + `metadata`), HuggingFace datasets exported to JSONL |
| SEC EDGAR submissions | `.txt` (content-detected) | 10-K, 10-Q, 8-K full-submission files |

To ingest a completely different dataset right now, just point the pipeline at it with a custom collection name:

```python
from src.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

# Any PDF — earnings transcript, research paper, central bank report
result = pipeline.ingest_file_sync(
    "data/raw/apple_earnings_q4_2025.pdf",
    collection="apple_earnings"        # ← your own isolated collection
)

# A whole folder of documents in any supported format
result = pipeline.ingest_directory_sync(
    "data/raw/equity_research/",
    collection="equity_research"
)

print(f"Indexed {result.total_chunks} chunks from {result.successful} files")
```

Each `collection=` name creates its own **isolated Qdrant collection + BM25 index + DocumentRegistry** hence datasets never contaminate each other.

---

### 7.2 Querying a Different Collection

`HybridRetriever` defaults to `collection_name="sec_filings"`. To query a different dataset, pass the collection name at instantiation. In `src/agents/langgraph_agent.py`, line 73:

```python
# Current default
retriever = HybridRetriever()

# Point at your own collection
retriever = HybridRetriever(collection_name="equity_research")
```

For a dynamic, per-request collection selection, extend `QueryRequest` in `src/main.py`:

```python
class QueryRequest(BaseModel):
    query: str
    collection: str = "sec_filings"   # optional override per request
```

Then pass `request.collection` when constructing the retriever or when calling `retriever.search()`.

---

### 7.3 Adding a New File Format

Adding support for a new format requires **one new file** nothing else changes. The pattern is:

1. Subclass `BaseParser` in `src/ingestion/parsers/`
2. Implement `supported_extensions()` and `parse()` both return a `ParsedDocument`
3. Register it in `ParserRegistry.default()`

**Example — CSV parser:**

```python
# src/ingestion/parsers/csv_parser.py
import csv
from typing import Set
from src.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedPage, ContentType

class CSVParser(BaseParser):
    def supported_extensions(self) -> Set[str]:
        return {".csv"}

    def parse(self, file_path: str) -> ParsedDocument:
        rows = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))

        full_text = "\n\n".join(rows)
        metadata = self._build_metadata(file_path, title=file_path)
        metadata.compute_hash(full_text)

        return ParsedDocument(
            content=full_text,
            metadata=metadata,
            pages=[ParsedPage(page_number=1, content=full_text,
                              content_type=ContentType.FINANCIAL)],
            sections=self._detect_sections(full_text),
        )
```

Register it in `src/ingestion/parsers/registry.py` inside `ParserRegistry.default()`:

```python
from src.ingestion.parsers.csv_parser import CSVParser
registry.register(CSVParser())
```

That's it. The chunker, embedder, Qdrant indexer, BM25 builder, and agent all inherit the new format automatically, no other code changes required.

The same pattern applies for: Excel (`.xlsx`), Word (`.docx`), EPUB, XML, database exports, etc.

---

### 7.4 Ingesting From APIs or Databases

For data that doesn't come from files (e.g., Bloomberg Terminal, Polygon.io, SQL databases, REST APIs), write a **fetcher** that produces a `ParsedDocument` and feeds it directly to the pipeline's embedding and indexing step:

```python
import asyncio
from src.ingestion.parsers.base import (
    ParsedDocument, DocumentMetadata, ParsedPage, ContentType
)
from src.ingestion.pipeline import IngestionPipeline

def fetch_fundamentals_from_api(ticker: str) -> ParsedDocument:
    """Fetch live data and wrap it in a ParsedDocument."""
    data = my_api_client.get_financials(ticker)   # your data source
    text = (
        f"{ticker} Financial Fundamentals\n\n"
        f"Revenue: {data['revenue']}\n"
        f"Net Income: {data['net_income']}\n"
        f"EPS: {data['eps']}\n"
        f"Date: {data['period']}"
    )
    meta = DocumentMetadata(
        source_path=f"api://{ticker}/fundamentals",
        source_name=f"{ticker}_fundamentals",
        file_type=".api",
        title=f"{ticker} Fundamentals — {data['period']}",
    )
    meta.compute_hash(text)
    return ParsedDocument(
        content=text,
        metadata=meta,
        pages=[ParsedPage(page_number=1, content=text,
                          content_type=ContentType.FINANCIAL)],
    )

# Feed directly into the pipeline
pipeline = IngestionPipeline()
doc = fetch_fundamentals_from_api("NVDA")
chunks = pipeline.chunker.chunk(doc)
asyncio.run(pipeline._embed_and_index(chunks, collection="live_fundamentals"))
```

---

### 7.5 What Is and Isn't SEC-Specific

| Component | SEC-specific? | Action needed for a new domain |
|---|---|---|
| `SECEdgarParser` | Finance/SEC only | None, it auto-detects SEC files via content; won't affect other formats |
| Default `collection_name="sec_filings"` | Default only | Pass your own `collection=` argument |
| Agent system prompts ("Senior Financial Strategist") | Partially | Optionally swap persona in `PLANNER_PROMPT` / `REASONER_PROMPT` |
| `HybridRetriever` | Fully generic | No changes needed |
| `SemanticChunker` | Fully generic | No changes needed |
| `IngestionPipeline` | Fully generic | No changes needed |
| `CollectionManager` / `DocumentRegistry` | Fully generic | No changes needed |
| `TokenBudgetManager`, `ModelRouter`, all caches | Fully generic | No changes needed |
| `RagasEvaluator` | Fully generic | No changes needed — evaluates any domain |
| `FastAPI` server (`main.py`) | Fully generic | No changes needed |

---

## 8. Limitations & Improvements

| Gap | Impact |
|---|---|
| **Single-process, in-memory caches** | `SemanticResponseCache` and `_EmbeddingCache` live in RAM. Multi-worker FastAPI (via Gunicorn) would have N separate cache instances with no sharing. |
| **Local Qdrant (file-based)** | Excellent for development but cannot be shared across multiple API server instances. A remote Qdrant server would be required for horizontal scaling. |
| **Sequential BM25 rebuild** | After batch ingestion, BM25 is rebuilt by scrolling _all_ Qdrant points **O(n)** with collection size. At 100k+ chunks this becomes slow. |
| **BM25 as pickle file** | The pickle deserialization loads the entire corpus into RAM. A 1M-document corpus would exhaust memory. |

#### Retrieval

| Gap | Impact |
|---|---|
| **No metadata filtering at search time** | The rich `section_path`, `content_type`, and `page_number` metadata in chunk payloads is not yet used for Qdrant payload filtering. Queries like "find only tables in Item 7A" would still scan all chunks. |
| **Fixed `fetch_k=50`** | The RRF candidate set is fixed at 50 per path, regardless of collection size. For very large collections this may miss relevant documents. |
| **BM25 tokenizer is naive** | `text.lower().split()`: No stemming, no stop word removal, no financial entity normalization. "revenue" and "revenues" are different tokens. |
| **No query expansion** | Sub-queries are not expanded with synonyms or financial term variants (e.g., "net income" → "net earnings", "profit"). |

#### Agent Architecture

| Gap | Impact |
|---|---|
| **No conversation history** | Each query is stateless. A user cannot follow up with "How does that compare to last year?" without repeating full context. |
| **Reflector uses truncated context** | `[:400]` chars per chunk in the reflection prompt — the reflector may miss details in long chunks, leading to false "needs_more_info" signals. |
| **Token estimation is approximate** | `len(text) * 0.25` (4 chars per token) can be significantly wrong for financial data with many numbers, symbols, and short tokens. Over-counting leads to premature tier escalation. |
| **No query routing** | All queries go through the full 5-node pipeline. Simple factual queries (e.g., "What does NVIDIA stand for?") that don't need multi-hop planning still execute the planner and rewriter. |

#### Evaluation

| Gap | Impact |
|---|---|
| **RAGAS test suite is 3 questions** | The hardcoded test suite (`ragas_evaluator.py` `__main__`) is too small to be statistically meaningful. |
| **No ground truth dataset** | Faithfulness and relevancy metrics require a curated QA dataset with reference answers for true precision/recall measurement. |
| **No regression testing** | There is no CI/CD pipeline that runs the evaluation suite on code changes. |

---


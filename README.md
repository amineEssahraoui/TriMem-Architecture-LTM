# TriMem: A Triple-Memory Architecture for Persistent Personalized AI Agents

## Overview
TriMem is a biologically-inspired, multi-layered memory architecture designed for conversational AI agents. It addresses one of the most critical limitations in modern LLM-based systems: the lack of persistent, personalized, and privacy-preserving memory across sessions, as well as the "stale memory" problem common in standard Retrieval-Augmented Generation (RAG) implementations.

Grounded in cognitive neuroscience models of human long-term memory, TriMem implements three specialized subsystems:
- **Episodic Memory:** Captures experiential context and conversation transcripts.
- **Semantic Memory:** Stores structured, up-to-date factual knowledge as a Knowledge Graph.
- **Procedural Memory:** Enforces user-defined behavioral rules and Standard Operating Procedures (SOPs).

---

## Key Features & Innovations

### 1. Ground Truth Hierarchy
TriMem resolves conflicts between memory types through a strict priority hierarchy during context assembly:
**Procedural Memory (Highest Priority) > Semantic Memory (Absolute Ground Truth) > Episodic Memory (Contextual Background)**
When a user updates a fact (e.g., changing a password or role), Semantic Memory overrides any outdated information still present in past Episodic conversation transcripts.

### 2. Cardinality-Aware Semantic Knowledge Graph
Built on SQLite, the semantic store extracts `Subject-Relation-Object` triplets constrained by a closed ontology of 17 relation types across 5 categories (Identity, Academic/Professional, Technical, Projects, Preferences). It handles relation cardinality natively:
- **1-to-1 Relations (Overwrite Model):** Attributes like `HAS_NAME`, `HAS_ROLE`, or `USES_OS` automatically overwrite previous values to prevent conflicting duplicates.
- **1-to-N Relations (Append Model):** Attributes like `HAS_SKILL` or `USES_TOOL` accumulate distinct entries while preventing exact duplicates.
- **Asymmetric Vector Retrieval:** Stored triplets are translated into natural language sentences and queried via vector similarity (`search_query:` vs. `search_document:` embeddings).

### 3. Dual-Storage Hybrid Episodic Retrieval & Temporal Decay
Every interaction is stored simultaneously in two backends:
- **ChromaDB:** HNSW cosine similarity search for semantic meaning.
- **SQLite FTS5:** Full-text keyword search (`OR` matching) for exact lexical recall.

Retrieved memories are deduplicated and ranked using an exponential temporal decay formula weighted by cognitive importance:
`S(m) = I_m * exp(-lambda * (t_now - t_m) / 86400)`
where `I_m` is the LLM-assigned importance score (1-10) and `lambda = 0.02` (approx. 34.7-day half-life).

### 4. Cognitive Importance Scoring Gate
Incoming messages are evaluated on a 1-10 cognitive scale across four tiers:
- **1-3 (Noise & Inquiry) & 4-5 (Ephemeral):** Stored in Episodic Memory only.
- **6-8 (Profiling) & 9-10 (Critical):** Trigger the full asynchronous extraction pipeline for Semantic and Procedural memories.
This gating mechanism filters out approximately 60% of routine conversational noise from polluting long-term factual and behavioral stores.

### 5. Hierarchical Procedural Memory (SOPs)
Behavioral rules are categorized and retrieved through a three-tier cascade:
1. **Personal Global SOPs:** Fetched unconditionally for every interaction (e.g., language preferences).
2. **Personal Task-Specific SOPs:** Retrieved via metadata-filtered vector search when the query matches a specific `task_tag` (e.g., coding guidelines).
3. **Universal SOPs:** System-wide fallback rules when no personal rule applies.

### 6. LLM-Driven Memory Consolidation
Triggered periodically every `N` interactions per user, the consolidation pipeline analyzes recent episodic records against the current Semantic Knowledge Graph. Redundant interactions (whose factual content is already captured in semantic storage) are pruned from ChromaDB, while nuanced interactions are preserved, achieving an average episodic compression ratio of 60%.

---

## System Architecture & Tech Stack

TriMem is structured into four distinct layers: Presentation/API, Core Services, Memory Subsystems, and Infrastructure.

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **API Framework** | FastAPI & Pydantic v2 | Asynchronous REST API (`POST /chat`) and schema validation |
| **LLM Inference** | Groq API (`llama-3.1-8b-instant`) | Response generation, importance scoring, fact/SOP extraction, consolidation |
| **Embedding Engine** | Ollama (`nomic-embed-text`) | Local 768-dimensional dense text embeddings (zero API cost/latency) |
| **Vector Database** | ChromaDB (`PersistentClient`) | Semantic storage for Episodic and Procedural memories |
| **Relational & FTS Store** | SQLite & SQLite FTS5 | Knowledge Graph triplet storage and full-text keyword indexing |
| **Visualization** | PyVis | Interactive Knowledge Graph rendering |
| **Language** | Python 3.11+ | Core implementation |

---

## Data Flow Lifecycle

1. **Synchronous Read Path:**
   - User sends a request to `POST /chat` with `user_id` and `message`.
   - The message is embedded locally using `nomic-embed-text`.
   - Relevant context is retrieved in parallel across Episodic (hybrid search + decay), Semantic (NL triplet cosine similarity >= 0.60), and Procedural (3-tier cascade) stores.
   - A structured system prompt enforcing the Ground Truth Hierarchy is assembled, and the LLM generates the response immediately (< 5s latency).

2. **Asynchronous Write Path (`BackgroundTasks`):**
   - The user message is scored by the Importance Scoring Engine (1-10).
   - The interaction is persisted in Episodic storage (ChromaDB + SQLite FTS5).
   - If the importance score is `>= 6`, Semantic fact extraction and Procedural SOP extraction are executed.
   - The per-user interaction counter increments; if it reaches `CONSOLIDATION_N`, the memory consolidation pipeline runs automatically.

---

## Project Structure

```text
app/
├── api/
│   └── routes.py            # FastAPI endpoints, context assembly, background task handlers
├── core/
│   ├── config.py            # Environment configuration and system constants
│   ├── embedding.py         # Ollama nomic-embed-text wrapper
│   └── llm.py               # Groq client, response generation, and importance scoring rubric
├── memory/
│   ├── episodic.py          # Dual-storage (ChromaDB + FTS5), hybrid retrieval, temporal decay
│   ├── semantic.py          # SQLite Knowledge Graph, 17-relation ontology, 1:1 / 1:N handling
│   ├── procedural.py        # 3-tier SOP classification and hierarchical vector retrieval
│   └── consolidation.py     # Periodic LLM-driven redundancy detection and pruning
└── tests/
    ├── test_globale_finale.py
    ├── test_scorer.py
    ├── test_semantic.py
    ├── test_procedural.py
    └── test_consolidation.py
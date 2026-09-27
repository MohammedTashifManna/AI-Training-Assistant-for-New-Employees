# AI Training Assistant for New Employees

An agentic onboarding assistant built with the **Google Agent Development Kit
(ADK)**. New joiners ask questions in natural language; a router agent
classifies each question and delegates it to one of four paths, and the
answering agent grounds its reply in retrieved company documents and cites them.

```
                          ┌───────────────────────────┐
   "How many days of      │      onboarding_router    │   root LlmAgent
    annual leave do   ───▶│      (the decision node)  │   classify + delegate
    I get?"               └─────────────┬─────────────┘
                                        │ transfer_to_agent
        ┌───────────────────┬───────────┴───────────┬────────────────────┐
        ▼                   ▼                       ▼                    ▼
 company_knowledge_   role_knowledge_        policy_admin_       general_support_
      agent                agent                 agent                agent
        │                   │                       │                    │
 search_company_     search_role_          search_policy_          (no tool:
   knowledge          knowledge              knowledge          direct answer)
        │                   │                       │
        ▼                   ▼                       ▼
   data/company/       data/roles/            data/admin/
   45 passages         59 passages            86 passages
```

| Route | Handles | Knowledge source |
| --- | --- | --- |
| `company` | What the company does, history, mission and values, offices, org structure, products, customers, acronyms | `data/company/` |
| `role` | Team charters, role expectations, tech stacks, grade ladder, promotion, performance, on-call | `data/roles/` |
| `admin` | Leave, expenses, travel, payroll, benefits, notice, conduct, security, training, system access | `data/admin/` |
| `general` | Generic technical or professional questions, questions about the assistant, small talk, out-of-scope | None — direct LLM answer |

> **Manna's Analytics is a fictional company.** Every document in `data/` is
> synthetic and was written for this project.

---

## Quick start

Four commands from a clean clone. No API key needed to get a working demo.

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 2. Build the retrieval index from the Markdown knowledge base
python scripts/build_index.py

# 3. Ask something
python scripts/ask.py "How many days of annual leave do I get?" --trace

# 4. Launch the chat UI
streamlit run ui/streamlit_app.py
```

### Requirements

- **Python 3.11+** (developed and verified on 3.13.2, Windows)
- No API key required for offline mode; a free
  [Google AI Studio key](https://aistudio.google.com/apikey) enables Gemini

---

## The two run modes

The project runs with or without credentials, and says which mode it is in
everywhere it matters.

| | **Offline mode** (default, no key) | **Live mode** (API key set) |
| --- | --- | --- |
| Routing | Real ADK router agent, backed by a deterministic model | Real ADK router agent, backed by Gemini |
| Retrieval | Local TF-IDF + BM25 hybrid search | Gemini embeddings + BM25 hybrid search |
| Answers | **Extracted** from retrieved passages | **Generated** by Gemini, grounded in passages |
| Network | None | Gemini API |
| Cost | Free | Gemini API usage |

Offline mode exists so the system is reproducible and reviewable without
credentials. The ADK agent graph, the router, `transfer_to_agent`, the retrieval
tools, session state and the UI are all genuinely exercised; only text
generation is swapped for a deterministic extractive backend
(`src/training_assistant/models/offline_llm.py`). Offline answers are always
tagged so an extract is never mistaken for generated prose.

### Enabling live mode

```bash
cp .env.example .env          # copy .env.example to .env on Windows
```

Set your key in `.env`:

```dotenv
GOOGLE_API_KEY=your-key-here
TRAINING_ASSISTANT_MODEL=gemini-3.5-flash
```

Then **rebuild the index**, because live mode uses Gemini embeddings and the two
index types are kept in separate files so they can never be mixed:

```bash
python scripts/build_index.py     # writes index/knowledge_index_gemini.json
```

> **Model naming note.** The Gemini 2.5 family is scheduled for retirement in
> October 2026, so this project defaults to `gemini-3.5-flash`. If your account
> does not have it, set `TRAINING_ASSISTANT_MODEL` to a model you do have.

---

## Usage

### Chat UI (primary interface)

```bash
streamlit run ui/streamlit_app.py
```

Every answer shows the route it took, an expandable **Sources** panel with each
retrieved passage, its relevance score and its file, and an expandable
**Routing trace**. Sample questions for each route are offered on first load.

### Command line

```bash
# One question
python scripts/ask.py "What is the mileage rate if I drive my own car?"

# With the full routing trace
python scripts/ask.py "Who is the CEO?" --trace

# Machine-readable
python scripts/ask.py "What is my notice period?" --json

# Interactive chat, keeps conversation context
python scripts/ask.py
#   commands: /trace  /reset  /quit
```

### ADK's own developer UI

Because the graph is exposed as a standard ADK agent package, ADK's built-in
inspector works too. It shows the raw event stream, every transfer and every
tool call.

```bash
adk web          # then choose "adk_app" in the agent dropdown
```

### Rebuilding and inspecting the index

```bash
python scripts/build_index.py              # build
python scripts/build_index.py --dry-run    # chunking report only, no embedding
python scripts/build_index.py --verbose
```

Rebuild whenever you edit anything in `data/`, change the embedding backend, or
change tokenisation in `knowledge/text_utils.py`.

### Evaluation

```bash
python scripts/evaluate_routing.py                    # everything
python scripts/evaluate_routing.py --rules-only       # fast, no agent runs
python scripts/evaluate_routing.py --json report.json
```

---

## Results

Measured in offline mode over 190 indexed passages. See
[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for method and analysis.

| Routing configuration | Labelled set (60 q) | Held-out paraphrases (25 q) |
| --- | --- | --- |
| Keyword classifier alone | 100.0% (macro F1 1.000) | 56.0% (macro F1 0.582) |
| **Keyword rules + retrieval probe** | **100.0%** (macro F1 1.000) | **80.0%** (macro F1 0.801) |
| Full agent pipeline | 100.0% (macro F1 1.000) | 80.0% (macro F1 0.801) |

**Grounding.** Of the questions routed to a retrieval path, 100% (71/71)
retrieved at least one passage *and* cited it in the answer. Mean 3.9 passages
per answer.

The two numbers per row matter more than either alone. The labelled set was
written alongside the routing rules, so 100% is an upper bound rather than a
generalisation estimate. The paraphrase set is deliberately casual
("*if i quit how long do i have to stick around*") and shows what happens to
keyword routing when real users type: it drops to 56%. Adding a retrieval-based
second stage recovers 24 points without costing anything on the labelled set.

---

## Project structure

```
.
├── data/                                   Synthetic knowledge base
│   ├── company/    6 docs   overview, mission, org, products, offices, glossary
│   ├── roles/      5 docs   engineering, data & AI, sales & CS, product/corporate, careers
│   ├── admin/      7 docs   expenses, leave, conduct, compliance, onboarding, HR, tools
│   └── eval/                labelled question set + held-out paraphrase probe
│
├── src/training_assistant/
│   ├── config.py                Single source of settings; resolves run mode
│   ├── prompts.py               Every system prompt, kept away from the wiring
│   ├── assistant.py             Public API: TrainingAssistant.ask() + trace capture
│   ├── logging_setup.py         Consistent logging, quiets known ADK notices
│   ├── agents/
│   │   ├── graph.py             The ADK agent graph (router + 4 specialists)
│   │   └── tools.py             One retrieval FunctionTool per knowledge domain
│   ├── routing/
│   │   ├── classifier.py        Deterministic keyword classifier (stage 1)
│   │   └── resolver.py          Two-stage resolution with a retrieval probe
│   ├── knowledge/
│   │   ├── documents.py         Heading-aware Markdown chunking
│   │   ├── text_utils.py        Shared tokenisation and stemming
│   │   ├── embeddings.py        Gemini and TF-IDF backends behind one interface
│   │   ├── vector_store.py      Domain-partitioned store, sparse JSON persistence
│   │   └── retriever.py         Hybrid semantic + BM25 search, scoped per domain
│   └── models/
│       └── offline_llm.py       Deterministic ADK BaseLlm for keyless operation
│
├── ui/streamlit_app.py          Chat interface with citations and routing trace
├── adk_app/agent.py             Entry point for `adk web`
├── scripts/
│   ├── build_index.py           Build the retrieval index
│   ├── ask.py                   CLI, single question or interactive
│   ├── evaluate_routing.py      Routing accuracy + grounding evaluation
│   └── generate_deck.py         Builds the presentation deck
├── docs/
│   ├── TECHNICAL_REPORT.md      Problem, architecture, prompts, results, limits
│   ├── AI_Training_Assistant_Deck.pptx
│   └── evaluation_report.json   Machine-readable evaluation output
├── index/                       Generated artefacts (gitignored)
├── requirements.txt
└── .env.example
```

---

## How it works

### 1. Routing is a real agent decision

The root agent has no tools of its own except ADK's `transfer_to_agent`. Its
prompt describes the four destinations by *subject matter* and requires exactly
one transfer. Specialists set `disallow_transfer_to_parent` and
`disallow_transfer_to_peers`, so each question is classified once and the chosen
agent is accountable for the answer. Cross-domain questions are handled in the
prompts: a specialist answers its own part and names the area that owns the rest.

### 2. Retrieval is scoped by the routing decision

Each specialist has exactly one tool, wired to one collection. The domain is not
a parameter the model can choose, so an admin question cannot reach a role
document. The fallback agent has **no tool at all**, which is what makes it a
genuine fallback rather than a fourth retriever.

### 3. Search blends meaning and exact terms

Each search combines cosine similarity over embeddings with Okapi BM25 over the
same tokenisation, weighted 0.65 / 0.35 by default. Dense-only retrieval misses
the precise identifiers a policy corpus is full of ("L5", "Sev-1", "26 weeks",
"INR 9,000"); BM25-only misses paraphrases such as "how much holiday do I get"
against a document that says "privilege leave entitlement".

Chunking is heading-aware, so a citation names a section
("*Leave and Time Off Policy > Carry-forward, encashment and expiry*") rather
than an anonymous fragment.

### 4. Not knowing is a supported outcome

Passages below a relevance floor are discarded. When nothing survives, the tool
returns `no_match` and the agent is instructed to say the documents do not cover
the question and point to a human channel. For onboarding, a confident wrong
answer about notice periods or expense caps costs far more than an honest
"I don't know, ask `#ask-people-ops`".

---

## Configuration

All settings are read from `.env` (see `.env.example`) or the environment.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | *(empty)* | Gemini key. Empty means offline mode. |
| `TRAINING_ASSISTANT_MODEL` | `gemini-3.5-flash` | Chat model for all agents |
| `TRAINING_ASSISTANT_EMBED_MODEL` | `gemini-embedding-001` | Embedding model (live mode) |
| `TRAINING_ASSISTANT_EMBED_DIM` | `768` | Embedding width (live mode only) |
| `TRAINING_ASSISTANT_TOP_K` | `4` | Passages returned per search |
| `TRAINING_ASSISTANT_SEMANTIC_WEIGHT` | `0.65` | Semantic share of the hybrid score |
| `TRAINING_ASSISTANT_MIN_SCORE` | `0.05` | Relevance floor below which nothing is returned |
| `TRAINING_ASSISTANT_FORCE_OFFLINE` | `false` | Force offline mode even with a key |
| `GOOGLE_GENAI_USE_VERTEXAI` | `false` | Use Vertex AI instead of AI Studio |

---

## Troubleshooting

**`No index found at index/knowledge_index_*.json`**
Run `python scripts/build_index.py`. The filename carries the embedding backend,
so switching between offline and live mode requires a rebuild.

**Answers look like bullet-pointed extracts rather than prose**
That is offline mode working as designed. Add `GOOGLE_API_KEY` to `.env` and
rebuild the index for Gemini-generated answers.

**`API key not valid` in live mode**
Check the key in `.env`, or set `TRAINING_ASSISTANT_FORCE_OFFLINE=true` to keep
working without the API. The assistant degrades gracefully rather than crashing:
it reports the failure and falls back to deterministic routing.

**`404` or `model not found` in live mode**
Your account may not have `gemini-3.5-flash`. Set
`TRAINING_ASSISTANT_MODEL` to a model you can access.

**Retrieval quality dropped after editing a document**
Rebuild the index. Also rebuild after changing anything in
`knowledge/text_utils.py`, since tokenisation determines the stored vocabulary.

**Streamlit shows a stale answer or stale sources**
Click **New conversation** in the sidebar to start a fresh ADK session.

---

## Deliverables

| Deliverable | Location |
| --- | --- |
| Working prototype | `src/`, `ui/`, `scripts/` |
| Chatbot UI | `ui/streamlit_app.py` |
| Routing, retrieval, generation code | `src/training_assistant/` |
| Synthetic knowledge base | `data/company/`, `data/roles/`, `data/admin/` |
| Synthetic question set | `data/eval/synthetic_questions.json`, `data/eval/paraphrase_probe.json` |
| Setup and usage instructions | this file |
| Technical report | `docs/TECHNICAL_REPORT.md` |
| Presentation deck | `docs/AI_Training_Assistant_Deck.pptx` |
| Evaluation results | `docs/evaluation_report.json`, `scripts/evaluate_routing.py` |

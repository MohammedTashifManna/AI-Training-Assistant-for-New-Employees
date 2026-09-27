# AI Training Assistant for New Employees

### Technical Report

**Project:** Generative AI and Agentic AI capstone
**Framework:** Google Agent Development Kit (ADK) 2.9.1, `google-genai` 2.24.0
**Runtime:** Python 3.13.2
**Organisation modelled:** Manna's Analytics (fictional, all data synthetic)

---

## 1. Problem statement and business impact

### 1.1 The problem

A new employee's first weeks are dominated by questions with short, factual,
already-documented answers:

- *How many days of annual leave do I get?*
- *What is the hotel cap for a work trip to Mumbai?*
- *When do new engineers join the on-call rotation?*
- *I think I clicked a phishing link. What do I do?*

Every one of those answers exists in writing. The difficulty is not that the
information is missing, it is that it is **fragmented across systems with no
shared entry point**: an HR portal, a policy wiki, a finance tool, an
engineering handbook, and a dozen Slack channels. A new joiner does not yet know
which of those owns which question, and often does not know the internal
vocabulary needed to search ("privilege leave", not "holiday").

So the question gets asked of a person instead. That produces four costs:

1. **Slower ramp.** Time spent searching, or waiting on a reply, is time not
   spent learning the actual job.
2. **Interrupted colleagues.** The same twenty questions are answered
   repeatedly by managers, buddies and HR partners.
3. **Inconsistent answers.** A colleague's recollection of the expense cap is
   not the expense policy. Wrong answers about caps, notice periods and claim
   windows cost employees money and time.
4. **Questions that never get asked.** New joiners routinely avoid asking
   things they think are obvious. Sensitive topics — harassment reporting,
   conflicts of interest, security incidents — are exactly the questions people
   least want to ask a human, and exactly the ones where a wrong assumption is
   most damaging.

### 1.2 Why an agent rather than a search box

Search returns documents; a new joiner needs an answer. But a single
retrieve-then-answer pipeline over one merged corpus behaves badly here, because
the corpus is not homogeneous:

- "What does the company do?" and "What is the expense cap?" need different
  documents, different tone, and different caution.
- Merging all documents into one index lets a leave question retrieve a passage
  from an engineering charter, because both contain the words "days" and "team".
- Policy answers carry consequences and need deadlines, approvers and escalation
  routes attached. Company-overview answers do not.

Routing first, then retrieving within a bounded collection, makes each answer
narrower, more accurate and easier to audit — and makes the system's behaviour
explainable to the people who own those policies.

### 1.3 Business impact

| Impact | Mechanism |
| --- | --- |
| Faster onboarding | Immediate answers to documented questions, at any hour and in any time zone |
| Fewer interruptions | Routine questions stop reaching managers, buddies and HR |
| Consistent, current answers | Every answer is grounded in the current document and names its section, policy version and review date |
| Lower risk on sensitive topics | Harassment, security and conflict-of-interest questions get the documented confidential route, including the routes that bypass the management chain |
| Documentation feedback loop | Questions that retrieve nothing identify gaps in the handbook |
| Auditability | Every answer names its source, so a policy owner can verify what employees are being told |

The design target is not to replace HR. It is to ensure a person is consulted
for judgement, not for lookup.

---

## 2. Architecture and routing design

### 2.1 System overview

```
 ┌─────────────┐      ┌──────────────────────────────────────────────┐
 │  Streamlit  │      │            TrainingAssistant                 │
 │  chat UI    │─────▶│  ADK Runner + session, trace capture,        │
 └─────────────┘      │  graceful degradation      (assistant.py)    │
 ┌─────────────┐      └───────────────────┬──────────────────────────┘
 │  CLI        │─────▶                    │
 └─────────────┘                          ▼
 ┌─────────────┐          ┌───────────────────────────────┐
 │  adk web    │─────▶    │     onboarding_router         │  root LlmAgent
 └─────────────┘          │     THE DECISION NODE         │  transfer only
                          └───────────────┬───────────────┘
                                          │ transfer_to_agent (exactly once)
        ┌──────────────────┬──────────────┴───────┬───────────────────────┐
        ▼                  ▼                      ▼                       ▼
 company_knowledge_  role_knowledge_        policy_admin_         general_support_
      agent               agent                  agent                 agent
        │                  │                      │                      │
        │ search_company_  │ search_role_         │ search_policy_       │ no tool
        │   knowledge      │   knowledge          │   knowledge          │
        ▼                  ▼                      ▼                      ▼
 ┌────────────────────────────────────────────────────────┐      direct answer
 │            Hybrid retriever (retriever.py)             │      from general
 │   0.65 × cosine(embeddings)  +  0.35 × BM25(lexical)   │      knowledge only
 └───────────────────────────┬────────────────────────────┘
                             ▼
        ┌────────────────────────────────────────────┐
        │  Domain-partitioned vector store            │
        │  company: 45 │ role: 59 │ admin: 86 passages│
        └────────────────────────────────────────────┘
```

### 2.2 The decision node

The root agent is an ADK `LlmAgent` with four `sub_agents` and no tools of its
own beyond ADK's `transfer_to_agent`. Its prompt (`prompts.ROUTER_INSTRUCTION`)
enumerates the four destinations by subject matter and constrains it to exactly
one transfer with no prose of its own.

Three design decisions shape this node:

**One hop, then answer.** Each specialist sets
`disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True`. A
question is therefore classified once and the chosen agent is accountable for
the reply. Allowing specialists to bounce questions between themselves would
make the routing decision unfalsifiable and latency unpredictable. Cross-domain
questions are handled in the prompts instead: each specialist answers its own
part and names the area that owns the rest.

**Route on subject, not vocabulary.** The prompt states explicitly that
"How do I get access to GitHub?" is an *administrative access* question, not an
engineering question. Where a question spans domains, the rule is to pick the
domain that owns *the decision the employee needs*: "How much of my learning
budget can I spend?" is administrative even though learning relates to career
growth.

**The fallback is a legitimate destination.** The prompt says so directly. A
router that treats "I don't know" as failure will force every question into a
knowledge base and manufacture a grounded-looking answer from an unrelated
passage.

### 2.3 Two-stage route resolution

In offline mode, and as a safety net in live mode, routing is resolved by
`routing/resolver.py` in two stages.

**Stage 1 — deterministic keyword classifier** (`routing/classifier.py`).
Weighted signals per route: multi-word phrases matched on word boundaries
(weight 3.0), single high-signal words matched by stem (weight 3.0), ordinary
domain terms matched by stem (weight 1.0), and conversational framings such as
"can you explain" (weight 0.9, counted at most once). A structural rule detects
acronym lookups ("what does *ARR* stand for") and routes them to the glossary.
Scores are normalised, and a confidence floor sends genuinely ambiguous
questions to the fallback.

**Stage 2 — retrieval probe.** Stage 1 is trusted in two situations: it chose a
knowledge route with confidence ≥ 0.5, or it chose the general route on
*positive* evidence (the question contained generic-topic signals such as
"in Python" or "SQL join"). Otherwise the question is put to all three
collections and routed to whichever retrieves best, provided the top score
reaches 0.16 and beats the runner-up by 0.015.

The distinction in stage 1 between **absence of evidence** and **evidence of
absence** is the load-bearing part, and Section 5.2 shows what happens without
it.

### 2.4 Why a retrieval probe exists

The keyword classifier scored 100% on the labelled question set and **48%** on a
held-out set of casual paraphrases. Almost every failure had the same shape: no
keyword matched, so the question fell through to the fallback path.

```
"i paid for a taxi myself, how do i get that money back"   -> general  (want admin)
"if i quit how long do i have to stick around"             -> general  (want admin)
"who runs this place"                                      -> general  (want company)
```

The tempting fix is to add "taxi", "quit" and "runs this place" to the keyword
lists. That is a treadmill: it repairs the sentences already seen and teaches
nothing about the ones not yet seen.

The structural insight is that **a vocabulary gap in the router says nothing
about whether the corpus can answer the question** — and the corpus is right
there. Retrieval already performs robust, IDF-weighted matching over the real
document vocabulary, which is exactly the signal the keyword lists lack. So when
the rules are unsure, the collections are asked directly and the best answerer
wins. Out-of-scope questions are protected by a relevance floor deliberately
higher than the retriever's own: questions about vector databases or the weather
score below it everywhere and correctly stay on the general path.

This costs two thresholds and one extra retrieval per uncertain question, rather
than a permanently growing keyword list.

### 2.5 Layering

| Layer | Module | Responsibility |
| --- | --- | --- |
| Presentation | `ui/streamlit_app.py`, `scripts/ask.py`, `adk_app/agent.py` | Chat, CLI, ADK inspector |
| Orchestration | `assistant.py` | Session lifecycle, trace capture, graceful degradation |
| Agents | `agents/graph.py`, `agents/tools.py` | The ADK graph and its tools |
| Routing | `routing/classifier.py`, `routing/resolver.py` | Explainable decision logic |
| Knowledge | `knowledge/*` | Chunking, embedding, storage, hybrid retrieval |
| Models | `models/offline_llm.py` | Deterministic ADK model backend |
| Configuration | `config.py`, `prompts.py` | Settings and every prompt, in one place each |

No module below the presentation layer imports Streamlit, and no module outside
`config.py` reads `os.environ`. Prompts live away from the wiring because they
are the part most often iterated on.

---

## 3. Knowledge sources and retrieval approach

### 3.1 The synthetic corpus

Eighteen Markdown documents, written for this project, structured the way a real
handbook is: YAML front matter carrying `doc_id`, `owner`, `policy_version` and
`last_reviewed`, then headings, prose and tables.

| Domain | Documents | Passages | Content |
| --- | --- | --- | --- |
| `company` | 6 | 45 | Overview, mission and values, org structure and leadership, products and customers, offices and hybrid working, glossary of acronyms |
| `role` | 5 | 59 | Engineering, Data & AI, Sales & Customer Success, Product/People/Finance roles, career framework and performance |
| `admin` | 7 | 86 | Expenses, leave, code of conduct, compliance and information security, onboarding checklist, HR/payroll/benefits, internal tools and access |
| **Total** | **18** | **190** | |

The corpus was written to contain the properties that make retrieval hard, so
the system is tested against realistic conditions: exact figures in tables
(`INR 9,000`), cross-references between documents, near-duplicate topics in
different domains (on-call appears in both the engineering charter and the leave
policy), and deliberately confusable vocabulary (`AR` means Architecture Review,
not accounts receivable).

Two labelled question sets support evaluation:

- `synthetic_questions.json` — 60 questions with expected routes
  (company 14, role 15, admin 21, general 10).
- `paraphrase_probe.json` — 25 casual, indirect paraphrases held back from
  development.

### 3.2 Heading-aware chunking

Chunking follows Markdown structure rather than a fixed character window
(`knowledge/documents.py`). Each chunk is one section, carrying its heading path,
so a citation reads:

> *Leave and Time Off Policy > Carry-forward, encashment and expiry*

Sections above 1,600 characters are split on paragraph boundaries; tables are
never split mid-row, because a row is the atomic unit of meaning in a policy
document. Chunks under 120 characters are merged forward so a bare heading is
never indexed alone.

The document title and heading path are prepended to the text before embedding.
This is what lets a section titled "Notice and approval" be found by a question
about *leave* notice, even though the section body never repeats "leave".

### 3.3 Hybrid retrieval

Every search blends two signals, weighted 0.65 / 0.35:

**Dense.** Cosine similarity between L2-normalised vectors. In live mode these
come from `gemini-embedding-001`; in offline mode from an exact-vocabulary
TF-IDF vectoriser with sublinear term frequency. Both sit behind one `Embedder`
interface.

**Lexical.** Okapi BM25 (`k1=1.5`, `b=0.75`) over the same tokenisation. BM25
scores are unbounded, so they are squashed with `s / (s + 6)` rather than
min-max normalised — min-max would force the best candidate to 1.0 even when
nothing relevant exists, defeating the relevance floor.

Neither signal is sufficient alone. Dense-only retrieval misses the precise
identifiers a policy corpus is full of ("L5", "Sev-1", "26 weeks", "INR 9,000").
BM25-only misses paraphrases: "how much holiday do I get" against a document
that says "privilege leave entitlement". Candidates are drawn from the union of
each signal's top results, so a passage that only one signal likes still reaches
the blend.

Tokenisation is shared between both halves (`knowledge/text_utils.py`) so their
scores are comparable: lowercasing, a conservative stop list, a two-stage
stemmer, and a small irregular-form map.

### 3.4 Storage

A domain-partitioned store with JSON persistence (`knowledge/vector_store.py`).
An off-the-shelf vector database was considered and rejected for this corpus:
190 passages do not need a service, the routing design requires *strictly
partitioned* collections so an admin question cannot reach a role passage, and a
dependency-free store keeps the project reproducible on any machine while making
the retrieval mechanics visible rather than hidden.

The store records its embedding backend and dimensionality, and the index
filename carries the backend (`knowledge_index_tfidf.json` /
`knowledge_index_gemini.json`), so incompatible vectors can never be silently
mixed. Vectors are written sparsely when density is below 30%, which keeps the
offline index at 0.43 MB despite a 2,230-dimension vocabulary.

### 3.5 Retrieval scoping as an access-control boundary

Each specialist has exactly one tool, hard-wired to one domain. The domain is
**not a parameter the model can choose**. This is what makes the router's
decision binding rather than advisory: a mis-scoped question cannot quietly
reach into another collection. The fallback agent has no tool at all, so it
cannot cite a company document even by accident.

---

## 4. Prompt design and fallback handling

### 4.1 Prompt architecture

| Prompt | Applies to | Purpose |
| --- | --- | --- |
| `GLOBAL_INSTRUCTION` | Whole tree | Identity, audience, tone, hard boundaries |
| `ROUTER_INSTRUCTION` | Root | Classification rules and the four destinations |
| `_GROUNDING_CONTRACT` | Shared by 3 specialists | How to search, cite, and admit ignorance |
| `COMPANY_/ROLE_/ADMIN_AGENT_INSTRUCTION` | Specialists | Domain scope plus domain-specific cautions |
| `GENERAL_AGENT_INSTRUCTION` | Fallback | Answer generally, never state a company fact |
| `AGENT_DESCRIPTIONS` | Read by the router | What each destination covers |

Four principles run through all of them.

**Role and boundary first.** Each agent is told what it owns and, explicitly,
what it does not. Telling the specialists they do not own the other domains is
what stops them answering from general world knowledge when retrieved context is
thin.

**Grounding is mandatory and checkable.** Retrieval results arrive as a numbered
list and the specialists must cite those numbers inline. A reviewer can verify
any sentence against a named section of a named document. The numbering format in
`retriever.format_context()` and the citation instruction in the prompt are a
deliberate contract.

**Admitting ignorance is a success state.** Every retrieval agent gets explicit
permission, and an explicit format, for "the documents do not cover this".

**Escalation is part of the answer.** Policy questions have consequences, so the
prompts require a human route for anything sensitive, time-limited or
irreversible.

### 4.2 Domain-specific prompt rules

The admin agent carries the most consequence and therefore the strictest rules:
quote limits and deadlines exactly and never round or convert currency; surface
the deadline even when the employee did not ask, because many of these processes
expire; name both the system and the approver; and for harassment,
discrimination, bribery, fraud or a security incident, state the reporting route
that bypasses the management chain and make clear that good-faith reporting is
protected.

The role agent is told to anchor answers to a specific grade rather than a vague
general standard, to give the documented process and then point to the person who
decides an individual case, and never to predict whether a specific person will
be promoted.

The company agent is told that business figures are point-in-time and must carry
their period, and that customer names are confidential unless the passage
explicitly marks the account as a public reference.

### 4.3 Layered fallback handling

Failure is handled at four levels, each degrading rather than collapsing.

**1. Nothing relevant retrieved.** Passages below the relevance floor are
discarded. If none survive, the tool returns `status: "no_match"` with an
instruction not to answer from general knowledge, and the agent says the
documents do not cover the question and names the owning channel.

**2. Question is out of scope.** The router sends it to the fallback agent, which
has no retrieval tool and is required to distinguish clearly between "here is a
general answer" and "this needs a company source I do not have". Blurring those
two is called out in the prompt as the worst available outcome on that path,
because the employee cannot tell which they received.

**3. Infrastructure failure.** A missing index returns `status: "index_missing"`
with usable guidance rather than raising. Semantic scoring failure falls back to
lexical-only retrieval. A model or credential failure produces an actionable
message, records the error in the trace, and labels the route using the
deterministic classifier so the UI still shows something truthful.

**4. Uncertain routing.** Covered by the retrieval probe in Section 2.4.

### 4.4 Offline mode as an honesty problem

Without credentials, an LLM project is unrunnable code: a reviewer cannot see
routing happen, cannot see retrieval ground an answer, and cannot reproduce a
result. `models/offline_llm.py` addresses this with a deterministic
`BaseLlm` implementation that speaks ADK's protocol — it emits
`transfer_to_agent` as the router, emits retrieval calls as a specialist, and
produces a final answer once tool results return. Every piece of orchestration
under test is the real thing; only language generation is swapped.

Its answers are **extractive**: it selects the sentences and table rows that best
answer the question and labels them with citations. Line selection weights
overlap by corpus term rarity and boosts numeric lines for quantitative
questions, which is what makes "Mumbai, Delhi NCR — INR 9,000" outrank the table
header "Location | Cap per night" (Section 5.3).

Because extracts are not generated prose, **every offline answer is tagged as
such**. Presenting an extract as a model-written answer would misrepresent what
the system did.

---

## 5. Results and engineering findings

### 5.1 Routing and grounding

Offline mode, 190 indexed passages. Reproduce with
`python scripts/evaluate_routing.py`.

| Routing configuration | Labelled set (60 q) | Held-out paraphrases (25 q) |
| --- | --- | --- |
| Keyword classifier alone | 100.0% (macro F1 1.000) | 56.0% (macro F1 0.582) |
| **Keyword rules + retrieval probe** | **100.0%** (macro F1 1.000) | **80.0%** (macro F1 0.801) |
| Full agent pipeline | 100.0% (macro F1 1.000) | 80.0% (macro F1 0.801) |

Per-route on the labelled set: precision, recall and F1 all 1.000 across
company (14), role (15), admin (21) and general (10).

**Grounding.** Of 71 questions routed to a retrieval path across both sets, 100%
retrieved at least one passage *and* cited it. Mean 3.9 passages per answer.
Median end-to-end latency 0.02 s offline, which is retrieval and extraction only.

**On the 100%.** The labelled set was authored alongside the routing rules, so it
is an upper bound and not a generalisation estimate. It is reported because it
demonstrates the intended taxonomy is self-consistent, not because it predicts
field performance.

**On the paraphrase set.** It was written once, in deliberately casual language,
and scored before any tuning: the keyword classifier got **48%**. It then
informed three rounds of *structural* fixes, described below, after which the
same set reached 80%. It is therefore no longer strictly held out, and the 80%
should be read as "after fixing the classes of bug the probe exposed" rather than
as a clean out-of-sample number. No probe question was ever added to the keyword
lists.

The most useful result is the gap itself. **Keyword routing loses 44 points when
users phrase questions naturally.** That is the finding that justifies an
LLM-based router in production; the deterministic path exists as a baseline, an
offline fallback and a safety net, not as the intended primary.

### 5.2 What the probe exposed

Each of these was a class of bug, not a single wrong answer.

**Absence of evidence treated as evidence of absence.** The first version of the
retrieval probe fired whenever the classifier said "general". That dragged
genuinely general questions into knowledge routes — "explain SQL joins" retrieves
*something* from the role documents — and dropped the labelled set from 100% to
86.7%. The fix distinguishes "general because generic-topic signals matched" from
"general because nothing matched". Only the latter is probed.

**Stemmer asymmetry.** Three separate failures came from a query and its
answering passage stemming differently: `locations` → `location` versus
`located` → `locat`; `offices` → `offic` versus `office` → `office`;
`bookings` → `booking` versus `booking` → `book`. The stemmer was restructured
into two ordered stages (plural normalisation, then derivational stripping) with
an explicit minimum-stem length. All 31 inflection pairs now collide correctly
while short words such as "rate" and "state" stay intact. An irregular-form map
was added for cases no suffix rule reaches, so "when do I get **paid**" matches
"**pay** date".

**Overlapping signals double-counting.** Three overlapping framings
("can you explain", "explain what", "explain how") each scored 3.0, so
"can you explain how promotions work here?" scored 6 for general against 3 for
role — a clear role question routed to the fallback. Framings were separated into
their own class, weighted 0.9, and counted at most once. The weight sits
deliberately *below* the ordinary term weight, so a single domain word outranks
any framing.

**Single-word "phrases" breaking on plurals.** Entries like `promotion` were
matched by regex, and `\bpromotion\b` does not match "promotions". Single-word
entries are now matched by stem while keeping their higher weight.

**Ambiguous bare terms.** "customer", "office" and "booking" were removed from
the term lists. Every function talks about customers; "office" appears in "home
office" (an expense matter) as often as "our offices" (a company matter); and
"booking" stems to "book", so "recommend a good book" routed to admin. Their
unambiguous senses are expressed as multi-word phrases instead.

### 5.3 Retrieval and extraction findings

**Function words with high IDF.** "without" and "through" were absent from the
stop list, rare enough in the corpus to earn high IDF, and caused
"how do I report harassment without going through my manager?" to match
*Expense Policy > Not reimbursable* at 0.487 lexical score. After extending the
stop list, the Code of Conduct's reporting sections rank first and second.

**Manufactured similarity from hash collisions.** An early TF-IDF implementation
gave unseen query terms a high synthetic IDF and folded the vocabulary into 768
hashed buckets. Together these produced a 0.235 cosine score for
"airspeed velocity of an unladen swallow", which defeats the relevance floor that
stops the assistant inventing answers. Both were removed: unseen terms
contribute nothing, and the vocabulary is indexed exactly with no collisions.
Off-topic questions now correctly return zero passages.

**Rarity-weighted line selection.** Extractive answering initially scored lines
by flat term overlap, so for "what is the hotel cap per night in Mumbai?" the
table *header* ("Location | Cap per night") outscored the row containing the
answer ("Mumbai, Delhi NCR | INR 9,000") — the header repeats the question's
common words while the answer carries the one rare word that matters. Weighting
overlap by corpus term rarity, plus a numeric-line boost for quantitative
questions, puts the correct row first.

**Hard-wrapped Markdown.** Physical lines are not quotable units; quoting them
produced fragments cut mid-sentence. Prose is now reflowed into paragraphs before
sentence splitting, and indented bullet continuations are joined to their bullet.

**Sibling sections are one answer.** "Where are our offices located?" retrieves
three sibling subsections (Bengaluru, Austin, Berlin), but only one tends to
share wording with the question, so the answer listed a single office. Passages
sharing a parent heading path are now recognised as parts of one enumerated
answer.

### 5.4 ADK integration findings

**Inter-agent transcripts arrive as user text.** When ADK transfers control, it
gives the receiving agent a transcript of what the previous agent did, delivered
as `role="user"` **text** parts wrapped in `<<<BEGIN_QUOTED_AGENT_CONTENT>>>`.
Taking "the last user message" as the question therefore sent the *transfer
receipt* to the retriever. The pipeline still appeared to work end to end while
returning passages about onboarding circles and MDM agents for a leave question.
Injected scaffolding is now filtered explicitly.

**Turn scoping matters for multi-turn.** Scanning the whole conversation for tool
responses made turn two reuse turn one's retrieval. The model now isolates the
current turn: the latest genuine user utterance, and only the parts after it.

**Session state persists across turns.** A question answered on the
no-retrieval fallback path still displayed the previous question's sources.
Retrieval provenance is now cleared through `state_delta` at the start of every
turn.

Each of these produced *plausible* wrong behaviour rather than an exception,
which is the characteristic failure mode of agent systems and the reason the
routing trace and source panel are part of the product rather than a debug view.

---

## 6. Limitations

### 6.1 Data

The corpus is synthetic, single-language (English), and describes one fictional
company. It is internally consistent by construction, so the system has not been
tested against genuinely contradictory documents, badly formatted exports,
scanned PDFs, or the version skew of a real handbook. At 190 passages it is also
small; retrieval behaviour at 100,000 passages is not evidenced here.

### 6.2 Evaluation

Both question sets are self-authored, and Section 5.1 explains why the labelled
100% is an upper bound and why the 80% probe figure is no longer strictly out of
sample. Neither set was validated by real new joiners.

More importantly, **routing accuracy and citation presence are not answer
correctness**. The evaluation confirms that a question reached the right
collection and that the answer cites retrieved passages. It does not verify that
the answer is factually right, complete, or free of a subtly wrong figure. There
is no faithfulness scoring, no human rating, and no adversarial testing for
prompt injection through document content.

### 6.3 Offline extraction quality

Extractive answers are blunt and imperfectly ordered. Two known cases:
"per diem for meals in India" surfaces caveat lines above the
"India | INR 2,000 per day" row, because the caveats contain the rare word
"diem" and the answer row does not; and for "what is my notice period", the
notice table ranks below a leave-policy cross-reference to notice periods.
Offline mode is a reproducibility and demonstration aid, not the intended
answering path.

### 6.4 Architecture

- **One hop, no reconsideration.** A misrouted question is answered by the wrong
  specialist rather than being reconsidered. The prompts mitigate this by having
  specialists name the owning area, but there is no automatic recovery.
- **Single-domain answers.** A question genuinely spanning two domains
  ("what happens to my leave balance when I'm promoted?") retrieves from one
  collection only.
- **No memory across sessions.** Each conversation starts fresh; the assistant
  does not know the employee's role, location, grade or start date, so it cannot
  personalise a leave entitlement or name the right team charter unprompted.
- **In-memory sessions.** Restarting the process loses conversation history.
- **No live system integration.** The assistant answers from documents. It cannot
  read an actual leave balance, expense claim status or access request.
- **English only.**
- **Deterministic router's confidence is not calibrated.** The normalised score is
  a useful relative signal, not a probability.

### 6.5 Operational

There is no authentication, rate limiting, cost control, audit log or PII
handling in the current build. Answers are only as current as the last index
build, and nothing detects that a source document has changed.

---

## 7. Future improvements

### 7.1 Near term

1. **Answer-quality evaluation.** Add a faithfulness check that verifies every
   claim is supported by a cited passage, plus a small human-rated set. This is
   the largest current gap: the system measures whether it *looked* in the right
   place, not whether it was *right*.
2. **Personalisation from the HR record.** Passing role, grade, location and
   start date into session state would let the assistant answer "how much leave
   do I have?" with the employee's own accrual, and route role questions to the
   right team charter without being told.
3. **Feedback capture.** A thumbs up/down per answer, stored with the route and
   the retrieved passages, turns live usage into training and evaluation data and
   surfaces documentation gaps.
4. **Retrieval-gap reporting.** Log every `no_match` and report the clusters
   weekly to document owners. Questions the handbook cannot answer are the
   handbook's backlog.
5. **Confidence surfaced in the UI.** Show when an answer rests on weak
   retrieval, so employees know when to verify with a human.

### 7.2 Medium term

6. **Cross-domain answers.** Let the router fan out to two specialists for
   genuinely spanning questions and compose a single reply, keeping the
   attribution per part.
7. **Reranking.** A cross-encoder over the top 20 candidates would fix the
   ordering problems in Section 5.3 more generally than heuristics can.
8. **Persistent sessions.** ADK's `DatabaseSessionService` in place of the
   in-memory service, so conversations survive restarts.
9. **Live system integration.** Read-only tools against the HR and expense
   systems, moving from "here is the policy" to "here is your balance".
10. **Answer caching.** Common questions are highly repetitive; caching by
    normalised question and index version would cut cost and latency materially.
11. **Multilingual support.** Multilingual embeddings and answering in the
    employee's preferred language.

### 7.3 Longer term

12. **Learned routing.** Once feedback data exists, train a classifier on real
    questions and compare it against both the LLM router and the keyword
    baseline. The evaluation harness already reports all configurations
    side by side.
13. **Proactive onboarding.** Drive from the 90-day checklist rather than
    waiting to be asked: prompt about insurance nomination before the 30-day
    deadline, or about tax declaration in the proof window.
14. **Escalation with context.** When the assistant cannot answer, open a ticket
    to the owning team pre-filled with the question and what was searched.
15. **Document-change awareness.** Watch source documents, rebuild incrementally,
    and flag answers whose supporting passage has since changed.

---

## 8. Conclusion

The system delivers the specified behaviour: employee questions in natural
language, routed through a decision node to company, role, administrative or
direct-LLM paths, answered with an LLM grounded in organisational documents, and
served through a working chat interface.

Three decisions did most of the work. **Partitioning retrieval by the routing
decision**, with the domain fixed in code rather than chosen by the model, makes
the router's choice binding and each answer auditable. **Treating "I don't know"
as a first-class outcome**, enforced by a relevance floor and explicit prompt
permission, is what makes the assistant safe to point at policy documents where a
confident wrong answer about a notice period costs someone real money. And
**resolving uncertain routes by asking the corpus instead of extending a keyword
list** turned a brittle 48% into 80% on natural phrasings, while making the
system's weakest component explainable rather than merely bigger.

The most transferable finding is methodological. Every significant bug in this
build produced *plausible* wrong behaviour rather than an error: a retriever
searching for a transfer receipt, a table header outranking the row beneath it,
a stemmer that separated "offices" from "office", a probe that confused silence
with a verdict. None would have been caught by an exception, and several were
invisible in the final answer. They were found by measuring routing and grounding
explicitly, and by making the trace visible in the product. For agent systems,
observability is not instrumentation added afterwards — it is the only way to
know the thing works at all.

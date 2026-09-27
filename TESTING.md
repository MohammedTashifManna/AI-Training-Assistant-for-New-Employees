# Testing and Evaluation

How to evaluate the AI Training Assistant across the three things that decide
whether it is fit to put in front of a new joiner:

| # | Concern | Question it answers | Suite |
| --- | --- | --- | --- |
| 1 | **Routing accuracy** | Did the question reach the right knowledge base? | `tests/test_router.py` |
| 2 | **Faithfulness** | Did the answer invent anything? | `tests/test_faithfulness.py` |
| 3 | **Latency** | Where does the time go? | `tests/test_latency.py` |

---

## Quick start

```bash
# from the project root
pip install -r requirements.txt
python scripts/build_index.py        # the suite needs an index to test against
python -m pytest                     # all three suites, ~5 seconds
```

Expected on a clean run:

```
145 passed, 1 skipped in 5.2s
```

The skip is deliberate: one latency assertion only applies in live mode and says
so.

---

## Running individual suites

```bash
python -m pytest tests/test_router.py -v          # 51 tests
python -m pytest tests/test_faithfulness.py -v    # 81 tests
python -m pytest tests/test_latency.py -v         # 14 tests

python -m pytest -m router                        # by marker
python -m pytest -m faithfulness
python -m pytest -m latency
python -m pytest -m "not slow"                    # skip the benchmarks

python -m pytest tests/test_router.py::test_question_routes_correctly  # one test
python -m pytest "tests/test_router.py::test_question_routes_correctly[R12]"  # one case
```

### Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--judge=auto` | `auto` | Gemini judge when an API key is present, deterministic checker otherwise |
| `--judge=gemini` | | Force the LLM judge; fails loudly if no key is configured |
| `--judge=grounding` | | Force the deterministic checker even when a key exists |
| `--latency-runs=N` | `3` | Repetitions per question in the benchmark |
| `--allow-live` | off | Permit the *pipeline* to call Gemini. Off by default so results are deterministic and free |

The suite pins the pipeline to offline mode unless `--allow-live` is passed.
Without that, whether the tests hit the network would depend on which shell they
were launched from, and results would not be reproducible.

---

## 1. Routing accuracy

`tests/test_router.py` runs the dataset through the ADK decision node and scores
three layers separately, because they can disagree and the disagreement is the
useful part:

| Layer | What it is |
| --- | --- |
| `classify()` | The deterministic keyword classifier alone. The cheap baseline. |
| `resolve_route()` | Keyword rules plus a retrieval probe when the rules are unsure. |
| Full pipeline | The real ADK graph: router agent, `transfer_to_agent`, specialist. |

### Metrics

Accuracy alone is misleading on a four-class problem, so the suite computes
**per-category precision, recall and F1** plus a confusion matrix, and asserts
each category individually. A router that sent everything to `admin` would score
25% accuracy but 100% recall on `admin`; only the per-class view exposes that.

Two metrics are specific to this design:

- **False fallback rate** — the share of *answerable* questions dumped onto the
  direct-LLM path. A router under pressure over-uses the fallback because it is
  never obviously wrong. It is wrong here: the answer was in the corpus.
- **Retrieval scoping** — knowledge routes must retrieve, and the fallback route
  must retrieve *nothing*. This is the architectural guarantee the whole design
  rests on: the fallback agent has no retrieval tool, so it cannot cite a
  company document.

### Thresholds

```python
MIN_OVERALL_ACCURACY        = 0.90
MIN_MACRO_F1                = 0.90
MIN_PER_CATEGORY_RECALL     = 0.80
MIN_PER_CATEGORY_PRECISION  = 0.80
MAX_FALSE_FALLBACK_RATE     = 0.10
```

Set below measured performance so ordinary noise does not fail the build but a
real regression does. They are assertions about the product, not descriptions of
today's run.

### Measured

```
accuracy  : 20/20 = 100.0%
macro F1  : 1.000
false fallback rate: 0.0%

route         prec  recall      F1  support
company      1.000   1.000   1.000        5
role         1.000   1.000   1.000        5
admin        1.000   1.000   1.000        5
general      1.000   1.000   1.000        5
```

All three layers score 100% on these 20 questions. **This is a small, curated
set and should not be read as a generalisation estimate.** The project's wider
evaluation (`python scripts/evaluate_routing.py`, 60 labelled questions plus 25
held-out paraphrases) is the honest measure: there, keyword routing alone drops
to 56% on casual phrasings while the two-stage resolver holds 80%.

### Beyond scoring

Two behavioural tests that catch bugs a metric cannot:

- `test_routing_is_deterministic_within_a_run` — the same question must route
  identically twice. Offline the pipeline is deterministic, so variation means a
  state leak between sessions.
- `test_conversation_history_does_not_leak_routing` — asks an admin question,
  then a role question in the same conversation, and asserts the second is
  re-routed and re-retrieved. Turn-scoping bugs in a multi-agent loop show up
  exactly here.

---

## 2. Faithfulness and hallucination

`tests/test_faithfulness.py` asks the only question that matters for a policy
assistant: **does the answer assert anything the sources do not support?**

A terse answer that quotes the leave policy correctly is a success. A fluent
answer that invents an expense cap is a failure that costs an employee money,
and it will read *better* than the correct one.

### What answers are judged against

The **union** of two contexts:

1. the ground-truth `reference_chunks` in `tests/eval_dataset.json`, copied
   verbatim from the corpus, and
2. the passages the retriever actually returned.

The union is deliberate. If retrieval fetched a different but equally valid
section than the one the dataset names, the answer is still grounded, and
scoring it as a hallucination would conflate a ranking difference with a
fabrication. Whether retrieval reached the *expected* document is a separate
concern, tested separately by `test_retrieval_reaches_expected_documents`.

### The two judges

**`GeminiJudge`** — the LLM-as-a-Judge. Decomposes the answer into atomic
claims and rules on each against the context, returning structured JSON with a
support flag, a severity, the supporting span, and a reason. Used when
`GOOGLE_API_KEY` is set. Its rubric states that plausibility is not support and
that general world knowledge is not support, and it marks any unsupported
figure, amount, deadline, name, system or channel as **critical**.

**`GroundingJudge`** — used when no key is present so the suite still runs.
**It is not an LLM judge** and does not attempt semantic entailment. It targets
the failure mode that actually matters: fabricated *specifics*. Every figure,
monetary amount, duration, percentage, Slack channel and email address in the
answer must appear in the context. That catches invented caps, wrong notice
periods and made-up channels. It will not catch a claim that reuses grounded
numbers but misstates their meaning — which is precisely why the Gemini judge
exists, and why the scorecard prints that caveat every run.

### The judge is itself tested

A test that can never fail is worse than no test. Two controls:

- `test_judge_detects_an_injected_hallucination` feeds the active judge a
  deliberately corrupted answer (`INR 47,500`, a 92-day window, an invented
  Slack channel) and **requires** it to be caught. If this passes silently, no
  other result in the file means anything.
- `test_judge_accepts_a_faithful_answer` is the control in the other direction,
  so a judge that rejects everything is also caught.

### What else is checked

| Test | Guards against |
| --- | --- |
| `test_reference_chunks_match_corpus` | Dataset drift. Reference text must still exist verbatim in `data/`, or every faithfulness result is silently meaningless |
| `test_answer_has_no_ungrounded_claims` | Fabricated specifics in retrieval-backed answers |
| `test_answer_contains_the_required_facts` | Faithful-but-useless answers that quote a neighbouring paragraph |
| `test_citations_resolve_to_retrieved_passages` | Dangling `[7]` citations that look like evidence and cannot be checked |
| `test_retrieval_reaches_expected_documents` | Answers grounded in the *wrong* document |
| `test_fallback_answers_assert_no_company_specifics` | The no-retrieval path quoting figures or channels. With no sources on screen, any figure there is invented |
| `test_absent_topics_are_not_invented` | Confabulation on the six adversarial probes |

### Adversarial probes

`hallucination_probes` in the dataset name six things that **do not exist** in
the corpus. The correct behaviour is to decline:

| Probe | Bait |
| --- | --- |
| H01 | A cycle-to-work scheme (wellness explicitly excludes bicycles) |
| H02 | A profit-share bonus (only a performance bonus exists) |
| H03 | Pet insurance |
| H04 | A Chief Marketing Officer (no CMO on the exec team) |
| H05 | Sabbatical after 2 years (the policy says 4) |
| H06 | An Iceland-specific hotel cap (only "Rest of world" applies) |

These assert that the answer does not *affirm* the fabrication, which is a much
stronger check than looking for the words "I don't know" — a real system phrases
refusal many different ways.

### Thresholds

```python
MIN_FAITHFULNESS_RATE   = 0.95
MAX_CRITICAL_UNGROUNDED = 0      # one invented expense cap is one too many
MAX_DANGLING_CITATIONS  = 0
```

### Measured

```
judge                 : grounding-checker  (deterministic, not an LLM)
answers judged        : 15
fully grounded        : 15/15 = 100.0%
claims checked        : 105
claims supported      : 105/105 = 100.0%
critical fabrications : 0
```

> **Read this honestly.** In offline mode the assistant composes answers by
> *extracting* lines from retrieved passages, so grounding is close to
> tautological — text copied from the context is trivially supported by it. The
> tests that carry real weight offline are the adversarial probes, the citation
> checks, the fallback-path checks and the judge self-test. The grounding rate
> becomes a meaningful measurement once Gemini is generating free-form prose:
>
> ```bash
> # in .env: GOOGLE_API_KEY=...
> python scripts/build_index.py                                  # Gemini embeddings
> python -m pytest tests/test_faithfulness.py --allow-live --judge=gemini
> ```

---

## 3. Latency

`tests/test_latency.py` decomposes each request using boundaries taken from the
ADK event stream.

| Phase | Boundary |
| --- | --- |
| `routing` | Question received → router's transfer decision (the router model call) |
| `retrieval` | Specialist's tool call → tool response (wall time, includes ADK dispatch) |
| `search` | Time inside the retriever itself, reported by the tool |
| `generation` | Tool response → final answer text |
| `overhead` | Whatever the three phases do not explain |

`overhead` is reported rather than absorbed. A decomposition that always sums
exactly to the total is usually one that is quietly discarding something.

Instrumentation lives in `PhaseTimings` (`src/training_assistant/assistant.py`)
and is available on every answer, not just in tests:

```python
result = assistant.ask("How many days of annual leave do I get?")
print(result.timings.format_line())
print(result.timings.to_dict())
```

### Cold start is measured on purpose

The first request in a process loads the index and builds BM25 statistics:
roughly 500 ms against about 5 ms afterwards. `test_cold_start_is_reported_separately`
clears the process-wide retriever cache so the measurement is genuinely cold and
independent of test ordering, then repopulates it. Every other latency test uses
the `warm_assistant` fixture. Benchmarking without a warmup measures disk I/O
once and reports it as routing latency forever.

### Measured (offline, 60 samples, `--latency-runs=3`)

```
Per phase, all routes (milliseconds)
  scope                      n      mean       p50       p95       max
  total                     60      20.2      20.3      27.5      34.8
  routing                   60      10.9      10.7      13.4      14.7
  retrieval                 60       1.2       1.1       2.1       8.8
  search                    60       0.6       0.4       0.8       7.5
  generation                60       3.7       3.6       5.6       7.7

Total latency by route (milliseconds)
  admin                     15      22.2      21.4      27.5      30.4
  company                   15      21.7      20.9      27.3      34.3
  general                   15      15.1      14.6      18.5      23.0
  role                      15      21.6      21.0      23.6      34.8

Phase share of total (mean)
  routing      56.1%
  retrieval     5.4%
  generation   18.2%
  unattributed 20.3%
```

Two things worth reading off this table:

- **Routing dominates offline at ~56%.** That is not the router being slow, it is
  the *retrieval probe* running inside routing: when the keyword rules are
  unsure, all three collections are searched to decide the route. The cost is
  visible precisely because the phase boundaries are honest about it.
- **The `general` route is the fastest** at 15 ms against ~22 ms, because it does
  no retrieval. `test_fallback_route_does_no_retrieval` asserts that its
  retrieval time is exactly zero, which is a performance check doubling as an
  architectural one.

Percentiles are used rather than means throughout: the mean hides the tail, and
the tail is what an employee experiences on the awkward question.

### Budgets

Separate budgets per mode, because offline bounds local compute while live is
dominated by Gemini round trips:

```python
OFFLINE_BUDGETS = {"total_p95": 0.60, "routing_p95": 0.30,
                   "retrieval_p95": 0.30, "generation_p95": 0.20}
LIVE_BUDGETS    = {"total_p95": 25.0, "routing_p95": 12.0,
                   "retrieval_p95": 8.0,  "generation_p95": 20.0}
```

Deliberately generous against measured performance. A flaky performance test
gets disabled, and a disabled test is worse than a loose one.

A machine-readable report is written to `docs/latency_report.json` on every run
that records samples.

---

## The evaluation dataset

`tests/eval_dataset.json` — 20 routing questions (5 per route) and 6 adversarial
probes.

```json
{
  "id": "R12",
  "question": "What is the nightly hotel cap for a work trip to Mumbai?",
  "category": "administrative",
  "expected_route": "admin",
  "expects_retrieval": true,
  "reference_chunks": [
    {
      "doc_id": "admin-expense-policy",
      "citation": "Expense and Reimbursement Policy > Travel > Accommodation",
      "text": "Nightly caps, excluding taxes:\n\n| Location | Cap per night |\n..."
    }
  ],
  "must_contain_facts": ["9,000"],
  "must_contain_any": [],
  "expected_doc_ids": ["admin-expense-policy"],
  "notes": ""
}
```

| Field | Purpose |
| --- | --- |
| `expected_route` | Ground truth for routing |
| `expects_retrieval` | Whether this question should reach a knowledge base at all |
| `reference_chunks` | Ground-truth context, **copied verbatim** from `data/` |
| `must_contain_facts` | **All** must appear in the answer. For single-valued answers such as a cap |
| `must_contain_any` | **At least one** must appear. For enumerations where several answers are equally correct |
| `expected_doc_ids` | Which source documents retrieval should reach |

### Extending it

1. Add an entry to `routing_questions` with `expected_route` and
   `expects_retrieval`.
2. For a retrieval question, find the passage that actually answers it:

   ```bash
   python -c "import sys; sys.path.insert(0,'src'); \
   from training_assistant.knowledge.retriever import load_retriever; \
   r=load_retriever(); \
   [print(f'{x.score:.3f} {x.chunk.doc_id} | {x.citation}') \
    for x in r.search('admin','your question here', top_k=5)]"
   ```

3. Copy that chunk's text into `reference_chunks` **verbatim**.
   `test_reference_chunks_match_corpus` will fail if it does not match, which is
   the point: it is what makes the stored copies safe to trust.
4. Keep the five-per-route balance, or relax
   `test_dataset_is_balanced_and_well_formed`.

### Both `must_contain_facts` forms exist for a reason

Writing this dataset surfaced two of my own mistakes, which the tests caught:

- R01 asks *what the company does*. I initially required the founding year
  because it sits in the same source chunk. It is not part of a correct answer to
  that question, so the requirement was wrong, not the system.
- R05 asks which customers may be named publicly. Four accounts qualify.
  Requiring one specific name was arbitrary, so it became `must_contain_any`.

If a required-fact test fails, check whether the fact is genuinely part of a
correct answer before changing the system.

---

## What these tests found

The suite is not decoration. Building it surfaced five real bugs, all of which
produced *plausible* wrong behaviour rather than an exception:

1. **`QUERY_NOISE` was matched before stemming.** "telling" survived filtering,
   stemmed to `tell`, and collided with "tell your manager" in the onboarding
   checklist — so *"how do I report harassment without telling my manager?"* was
   answered with laptop and access instructions. A safety-critical question
   answered from the wrong document.
2. **BM25 length normalisation punished summary tables.** At the conventional
   `b=0.75`, the "Entitlements at a glance" table holding the headline
   *22 working days* ranked below two shorter sections that merely mentioned
   privilege leave. Lowering it to `0.6` fixed that with no regression across a
   sweep of 11 fact-lookup queries.
3. **Enumerations split across sibling chunks lost members.** *"Which offices
   does the company have?"* named two of three, because the Bengaluru chunk is
   the longest sibling and fell outside `top_k`. Fixed with sibling completion at
   retrieval time.
4. **The answer composer's sibling rule was too broad at depth 1**, treating
   every section of a document as a sibling and crowding out the line that held
   the answer.
5. **The cold-start test passed alone and failed in the suite**, because earlier
   tests had already warmed the process-wide cache. A test whose result depends
   on execution order is not measuring what it claims to.

---

## Troubleshooting

**`SKIPPED ... No knowledge index`**
Run `python scripts/build_index.py`. The index filename carries the embedding
backend, so switching between offline and live mode needs a rebuild.

**`Failed: --judge=gemini requires GOOGLE_API_KEY`**
Intentional. The LLM judge fails loudly rather than silently degrading to the
deterministic checker, because a quiet downgrade would misrepresent what was
measured.

**Latency budget failures on a shared or throttled machine**
Raise `OFFLINE_BUDGETS` in `tests/test_latency.py`, or run
`python -m pytest -m "not slow"` to skip the benchmarks.

**`test_reference_chunks_match_corpus` fails**
A document in `data/` changed. Re-extract the affected `reference_chunks` using
the snippet above. Do not delete the test; it is what keeps the dataset honest.

**Tests are slower than ~5 seconds**
The pipeline runs once per session and is shared across suites via the
`pipeline_results` fixture. If you add a test that constructs its own assistant
per case, you will pay full cost per test.

---

## Files

```
tests/
├── conftest.py            Fixtures, LatencyRecorder, terminal reporting
├── eval_dataset.json      20 routing questions + 6 adversarial probes
├── test_router.py         51 tests  routing accuracy, precision, recall
├── test_faithfulness.py   81 tests  LLM-as-a-Judge, grounding, hallucination
└── test_latency.py        14 tests  per-phase benchmarks
pytest.ini                 Markers, warning filters, test discovery
docs/latency_report.json   Generated: per-phase samples and percentiles
```

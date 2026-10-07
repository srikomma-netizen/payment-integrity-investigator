# Payment Integrity Investigator

A runnable agentic investigation workflow
for healthcare payment integrity: suspicious claims, duplicate payments,
coding anomalies, and vendor payment diversion. It shows how an LLM is used
*inside* a controlled workflow, never as the decision maker.

Everything is synthetic. No real PHI.

```
          upstream controls                         agentic investigation (LangGraph)
 ┌──────────────────────────┐    flagged    ┌──────────────────────────────────────────────────┐
 │ risk rules / models      │ ───────────▶  │ intake ─▶ supervisor ─┬─▶ gather_evidence (MCP tools, masked)
 │ R1 duplicate payment     │               │             ▲         ├─▶ assess_risk                      
 │ R2 amount outlier        │               │             └─────────┼─▶ retrieve_policy (hybrid RAG)     
 │ R3 unbundling            │               │                       └─▶ summarize (Claude, structured)   
 │ R4 prior confirmed case  │               │                              ▼                              
 │ R5 unverified bank change│               │                           verify ─▶ route ─┬─▶ human_review (interrupt)
 └──────────────────────────┘               │                                            └─▶ finalize    
                                            └──────────────────────────────────────────────────┘
                                                        │ re-association by role, outside the model
                                                        ▼
                                              investigator UI / API
```

## What is in here

| Path | What it shows |
|---|---|
| `investigator/security/phi.py` | PHI/PII de-identification before the model, HMAC tokens, vault, role-scoped re-association, leak check, audit |
| `investigator/tools/registry.py`, `mcp_server.py` | MCP-shaped tool server: masked at the boundary, per-role permissions, typed failures, audit; stdio MCP exposure |
| `investigator/risk/rules.py` | Upstream deterministic controls that flag cases with typed signals and evidence ids |
| `investigator/rag/ingest.py`, `index.py` | Structure-aware chunking with metadata; BM25 + vector fusion, metadata filtering, re-ranking, source validation, context expansion |
| `investigator/graph/` | LangGraph `StateGraph`: planner-executor supervisor, retries, step limit, checkpointer, `interrupt` for human review, resume by thread |
| `investigator/llm/` | Claude via the official SDK with Pydantic structured outputs, and a grounded deterministic fake behind the same interface |
| `investigator/evals/` | Golden set scored per stage: retrieval (recall, precision, MRR, authorized sources), generation (faithfulness, risk agreement, indicator recall), workflow (routing, degradation), with failure attribution |
| `investigator/feedback/store.py` | Investigator dispositions stored with the overridden recommendation; false-positive rate by rule, attribution, golden candidates |
| `api/main.py` | FastAPI: start case, role-scoped view, decision (resume), feedback summary, tool specs |
| `data/policies/` | Six policy documents including a superseded version, a draft, and a restricted policy, to prove source validation |
| `tests/` | 29 offline tests |

## Run it

```bash
pip install -r requirements.txt
python scripts/demo.py                        # offline walkthrough
python -m investigator.evals.run_evals        # stage-level eval report
python -m pytest -q
uvicorn api.main:app --reload                 # POST /cases, GET /cases/{id}?role=, POST /cases/{id}/decision
python -m investigator.tools.mcp_server       # tools over MCP (stdio)
```

Set `ANTHROPIC_API_KEY` to use Claude (`claude-opus-5-5`, override with
`INVESTIGATOR_MODEL`). The graph, guards, and evals do not change.

## Design choices

1. **The agent does not decide what is suspicious.** Rules and models run
   first and emit typed signals with evidence ids. The agent investigates and
   explains flagged cases, and can only cite ids that exist.
2. **PHI never reaches the model.** Tools mask at the boundary with
   deterministic tokens, so the model can still reason about "the same
   patient". Re-association happens in the application layer per role, with
   every resolution audited. The verify node runs a leak check on the output.
3. **Retrieval is validated, not trusted.** Chunks carry status, version,
   effective date, access level, and claim type. Superseded, draft, and
   restricted policy text is filtered before fusion and checked again before
   it reaches the model. Expansion adds the parent section so the model sees
   a coherent rule, not a fragment.
4. **Faithfulness is checked by code.** Every evidence id and policy citation
   in the summary is checked against what was actually gathered and retrieved.
   An unfaithful summary is routed to a human with the reason.
5. **Human review is a graph interrupt.** High tier, low confidence, missing
   evidence, or failed verification pauses the run at a checkpoint; the
   reviewer's decision resumes it by thread id and lands in the audit trail.
6. **Evals attribute failures to a stage.** Retrieval, generation, and routing
   are scored independently, so a regression names the stage. Overrides from
   investigators become golden candidates, not online model updates.


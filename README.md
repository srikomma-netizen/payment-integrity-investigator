# payment-integrity-investigator

An LLM-assisted workflow for investigating flagged healthcare claims: duplicate
payments, upcoding, unbundling, and vendor bank-detail changes.

The idea I wanted to try out: rules decide *what* is suspicious, the model only
helps investigate and explain it, and a person makes the call on anything
risky. PHI never goes to the model.

All data is synthetic. No real members, providers or PHI.

```
risk rules (R1-R5) flag a claim
        |
        v
intake -> supervisor --> gather_evidence   (tools; records masked before the model sees them)
             ^      \--> assess_risk
             |       \-> retrieve_policy   (hybrid search + checks on policy version/status)
             +--------/
        then: summarize -> verify -> route --> human_review (pause until someone decides)
                                          \--> finalize
```

## Running it

Run these from the repo folder. The `uvicorn` and `scripts/` commands need it; the `python -m` ones work anywhere once the package is installed.

```bash
pip install -r requirements.txt
pip install -e .                            # so the python -m commands work from any folder
python scripts/demo.py                      # runs a few cases end to end
python -m investigator.evals.run_evals      # eval report
python -m pytest -q
uvicorn api.main:app --reload               # POST /cases, GET /cases/{id}?role=, POST /cases/{id}/decision
python -m investigator.tools.mcp_server     # same tools over MCP (stdio)
```

Without an API key it runs against a scripted stand-in model, which is what the
tests use. Set `GEMINI_API_KEY` to use Gemini (`GEMINI_MODEL`, default
`gemini-2.5-flash`), or `ANTHROPIC_API_KEY` to use Claude (` `). Gemini
wins if both are set; `LLM_PROVIDER=gemini|anthropic|fake` forces one.

## Layout

```
investigator/
  synthetic.py       generated claims, payments, providers, vendors, prior cases
  risk/rules.py      R1 duplicate payment, R2 amount outlier, R3 unbundling,
                     R4 prior confirmed case, R5 unverified bank change
  security/phi.py    masking into tokens, and resolving them back per role
  tools/             tool registry with per-role permissions, plus an MCP server
  rag/               policy chunking and hybrid retrieval
  graph/             the LangGraph workflow
  llm/               Claude adapter and the stand-in
  evals/             golden cases and scoring
  feedback/          stores investigator decisions for later analysis
api/main.py          FastAPI service
data/policies/       six policy docs, including a superseded version, a draft and a restricted one
```

## Notes on a few choices

- **Rules flag, the model explains.** Rule output is typed and carries evidence
  ids, so the summary can only point at records that actually exist.
- **PHI is masked inside the tools**, so there's no code path that hands raw
  records to the model. Tokens are HMAC-based, so the same member gets the same
  token across claims. Resolving a token back to a name happens in the app,
  only for roles cleared for it, and every lookup is logged.
- **Retrieval checks the source, not just the match.** Superseded versions,
  drafts, not-yet-effective and restricted policies are dropped with a reason.
  They're checked again right before anything goes to the model.
- **The summary is checked by code.** Every evidence id and citation has to
  exist, and the text is scanned for PHI. If anything fails, the case goes to
  a person.
- **Human review is a LangGraph `interrupt`.** The run is checkpointed and
  resumed by case id with the reviewer's decision.
- **Tool failures don't crash the run.** They show up as missing evidence,
  confidence drops, and the case is routed to a person.
- **Evals score retrieval, generation and routing separately**, so a failing
  case says which stage broke.
- **Investigator decisions are stored with what they overrode.** That gives a
  false-positive rate per rule and candidates for the golden set. Nothing gets
  fed back into the model automatically.

## Things I'd do next

- Real embeddings and a cross-encoder reranker (the interfaces are there; right now it's hashed n-grams and word overlap).
- Postgres checkpointer so paused cases survive a restart.
- Get the caller's role from their identity instead of a parameter in the MCP server.

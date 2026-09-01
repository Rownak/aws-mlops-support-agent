## Phase 1 — Prove the seam

- [ ] **1.1 Scaffold `packages/scifact_rag/`**
  Minimal second project: `config.yml`, a `DocSource` for the SciFact dataset, ingest + ask. **No**
  agent, **no** Jira, **no** UI — this exists to prove `rag_core` is genuinely reusable.
  *Done when:* it ingests and answers a question using only `rag_core` + its own source adapter.

- [ ] **1.2 Fold back what 1.1 taught us**
  Anything `scifact_rag` had to work around becomes a `rag_core` fix. Expect a few — that is the
  design working, not failing.
  *Done when:* the workarounds are gone and both projects use `rag_core` unmodified.
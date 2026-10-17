# Decodability and erasure helpers

Shared helpers used by `scripts/concept_audit_v3/baselines.py`: concept definitions, frozen encoding, node sampling, streaming Ridge sufficient statistics and prediction metrics. Ridge covariance is accumulated as `X.T @ X`.

The historical phase-one erasure orchestrator remains available for inspection. Its saved experiment output is absent from the current workspace; the progress report does not validate historical erasure claims. Use the root README and the v3 runner for the reviewed workflow. Legacy default paths are not evidence of training provenance or split independence.

# BigTrack Policies

This package joins predictors and matchers and owns lifecycle decisions.

- `_decision.py`: candidate/decision/counter records plus score and box-agreement math.
- `base.py`: shared initialize/restore/update/template/debug/reset/close flow.
- `simple.py`: accepts the first result, remains active, and never updates templates.
- `score_gated.py`: one-candidate score/geometry gating, prediction fallback, counters, loss progression, and interval-based template permission.
- `__init__.py`: concrete policy exports.

Important current limits: policies generate one candidate; matcher metadata is not given to `decide`; score-gated recovery cannot accept visual evidence after entering `RECOVERY`; clipped-template configuration is not enforced.

See [`docs/architecture.md`](../../docs/architecture.md) for exact behavior and [`docs/development.md#adding-a-bigtrack-policy`](../../docs/development.md#adding-a-bigtrack-policy) before adding a policy.

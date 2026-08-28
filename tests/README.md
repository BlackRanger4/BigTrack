# Tests

The suite uses `unittest` classes and is compatible with pytest discovery.

```powershell
python -m unittest discover -s tests -v
python -m pytest
```

Policy tests use fake predictor/matcher objects. Neural matcher tests inject fake backends so default runs do not require weights. Real-checkpoint cases are opt-in through per-matcher environment flags. Predictor evaluation is deterministic across fixed scenarios and seeds.

See [`docs/testing-and-tools.md`](../docs/testing-and-tools.md) for the responsibility of every test module and [`docs/development.md`](../docs/development.md) for required coverage when adding a policy, matcher, predictor, contract, or tool.

Keep tests isolated from `ignores/` except explicitly enabled smoke tests. New default tests must not download assets, open GUIs, or depend on a developer-specific device/path.

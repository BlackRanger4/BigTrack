# Tools

- `run_bigtrack_fulltest.py`: import-safe launcher for the full tracking setup UI.
- `bigtrack_fulltest/`: component selection/configuration, video/folder frame sources, interactive runner/debug views, timings, and JSONL logging. See its [local guide](bigtrack_fulltest/README.md).
- `predictor_trajectory_ui.py`: seeded trajectory generation, multi-horizon predictor precomputation, visualization, and error details.

Launch from the repository root:

```powershell
python tools\run_bigtrack_fulltest.py
python tools\predictor_trajectory_ui.py
```

Tool registries are independent. New policies/matchers/predictors must be registered where relevant, tested outside GUI creation, and documented in [`docs/testing-and-tools.md`](../docs/testing-and-tools.md).

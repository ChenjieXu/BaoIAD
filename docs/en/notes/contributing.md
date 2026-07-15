# Contributing to BaoIAD

## How to Add a New Method

Adding a new method to BaoIAD involves several steps. Follow this checklist to ensure completeness.

### Checklist

1. **Implement the detector** — Create a new detector module under `baoiad/models/detectors/` and register it with the `MODELS` registry in `baoiad/registry.py`.

2. **Create configs** — Add config files under `configs/<method>/`:
   - `configs/<method>/<method>_<backbone>_<resolution>_mvtec_strict.py` — MVTec AD strict config
   - `configs/<method>/<method>_<backbone>_<resolution>_visa.py` — VisA unified config
   - `configs/<method>/README.md` — Config README (see requirements below)

3. **Update `baoiad/method_inventory.py`** — Add a `MethodEntry` to the `METHODS` tuple with the correct slug, display name, family, config paths, readme path, and alignment path.

4. **Create alignment evidence** — Add `docs/alignment/<method>.md` documenting the reference freeze, code-path parity, behavior probes, and benchmark stop-line (see requirements below).

5. **Add tests** — Add unit tests under `tests/` covering the new detector's forward pass, loss computation, and any custom components.

6. **Update documentation** — Update relevant docs pages (method zoo, FAQ, etc.) to reflect the new method.

## Code Style and Conventions

BaoIAD follows the **MMEngine** framework conventions:

- **Registry pattern**: All models, datasets, transforms, metrics, and hooks must be registered using the scoped registries in [`baoiad/registry.py`](../../../baoiad/registry.py) (scope `'baoiad'`).
- **Config-driven**: Every experiment is defined by an MMEngine config file. Use `_base_` inheritance from `configs/_base_/` where possible.
- **Detector interface**: Detectors should subclass `BaseDetector` from `baoiad.models.detectors.base` and implement `forward_train()` and `forward_test()`.
- **Type annotations**: Use Python type hints for function signatures.
- **Imports**: Use absolute imports from `baoiad.*`.

## Testing

Run the test suite with:

```bash
pytest tests/
```

For a specific test file:

```bash
pytest tests/test_models/test_detectors/test_patchcore.py
```

Tests should cover at minimum:
- Forward pass with dummy input
- Loss computation (for methods that train)
- Memory bank construction (for feature-memory methods)
- Output shape and dtype correctness

## Method Inventory Update Process

The method inventory in [`baoiad/method_inventory.py`](../../../baoiad/method_inventory.py) is the single source of truth for all methods in the benchmark. When adding a method:

1. Choose a unique, lowercase `slug` (e.g., `patchcore`, `efficientad`).
2. Set the `display` name to the official paper name.
3. Assign the correct `family` from the existing families in the inventory.
4. List all `config_paths` as relative paths from the repo root.
5. Set `readme_path` and `alignment_path` to the corresponding files.
6. Verify the entry by running:
   ```bash
   python -c "from baoiad.method_inventory import METHODS_BY_SLUG; print(len(METHODS_BY_SLUG))"
   ```

## Config README Requirements

Each method's `configs/<method>/README.md` should include:

- **Method name and reference**: Paper title, authors, year, and a link to the original paper or code.
- **Config summary**: What each config file does (strict vs. unified, MVTec vs. VisA).
- **Hyperparameters**: Key settings and their values (backbone, resolution, batch size, epochs, learning rate).
- **Usage**: Example commands for training and testing.
- **Expected results**: Reference metric values for MVTec AD and VisA (if available).

## Alignment Evidence Requirements

Each method's `docs/alignment/<method>.md` should document:

- **Reference freeze**: The exact source (repo URL, commit hash, version tag) used as the ground truth for migration.
- **Code-path parity check**: A statement confirming that the BaoIAD forward/backward paths match the reference implementation, with notes on any intentional deviations.
- **Behavior probes**: Numerical comparisons on intermediate outputs (e.g., feature statistics, loss values) between BaoIAD and the reference, confirming they match within tolerance.
- **Benchmark stop-line**: The final metric numbers on MVTec AD and/or VisA that serve as the alignment target.

## PR Submission Process

1. **Fork and branch**: Create a feature branch from `master`.
2. **Implement**: Follow the checklist above.
3. **Test locally**: Run `pytest tests/` and verify your method trains and evaluates without errors.
4. **Format**: Run `ruff check --fix baoiad/` to ensure code style compliance.
5. **Document**: Make sure all config READMEs, alignment evidence, and the method inventory are updated.
6. **Submit**: Open a pull request with a clear description of the method, its family, and any alignment notes.

### PR Review Criteria

- [ ] Detector registered under `baoiad.MODELS`
- [ ] Config files for MVTec AD (strict) and VisA
- [ ] Method inventory entry added
- [ ] Config README complete
- [ ] Alignment evidence provided
- [ ] Tests passing (`pytest tests/`)
- [ ] No regressions in existing methods

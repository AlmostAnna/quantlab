## Notebook Hygiene

All notebooks should have outputs and metadata stripped before committing.

### Automated (preferred)
We use [`pre-commit`](https://pre-commit.com/) with `nbstripout`. Install it once:
```bash
pip install pre-commit && pre-commit install
```
### Manual
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/example.ipynb
```

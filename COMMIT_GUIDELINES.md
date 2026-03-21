# Commit Guidelines — EEAI CA1

## Team Members

| # | Name | GitHub Username | Email |
|---|---|---|---|
| 1 | Hari Krishnan | Hari-K-Dev | harikrishnankvp@gmail.com |
| 2 | Pushkar Upadhyay | pushkarupadhyay | newpusupa023@gmail.com |
| 3 | Harshwardhan Dongare | harshwardhan003 | harshwardhan.dongare@gmail.com |

## Commit Ownership by Task

### Hari Krishnan (Teammate 3 — Code: Modelling & Controller)
Commits related to:
- `main.py` — controller logic, chained pipeline
- `modelling/modelling.py` — model orchestration, chained multi-output logic
- `model/randomforest.py` — chained training/prediction
- `model/base.py` — abstract base updates
- Git repo setup & integration testing

### Pushkar Upadhyay (Teammate 2 — Code: Data Pipeline + Doc Tables)
Commits related to:
- `preprocess.py` — data loading, deduplication, noise removal, translation
- `embeddings.py` — TF-IDF vectorization
- `modelling/data_model.py` — Data class, train/test split, rare class filtering
- `Config.py` — config updates
- `utils.py` — shared utilities
- Architecture tables (components, connectors, data elements) in doc

### Harshwardhan Dongare (Teammate 1 — Architecture Diagrams & Doc)
Commits related to:
- Architecture sketch for Design Choice 1 (Chained Multi-Outputs)
- Architecture sketch for Design Choice 2 (Hierarchical Modelling)
- `docs/` folder — diagrams, doc drafts
- Final .doc compilation and submission

## Commit Schedule (March 2026)

| Date Range | Milestone | Who |
|---|---|---|
| Mar 1–3 | Repo setup, skeleton code, Config.py | Hari |
| Mar 4–6 | Preprocessing: data loading, deduplication | Pushkar |
| Mar 7–9 | Preprocessing: noise removal, translation | Pushkar |
| Mar 10–12 | Embeddings (TF-IDF), Data class | Pushkar |
| Mar 13–15 | Base model updates, initial RandomForest | Hari |
| Mar 16–18 | Chained multi-output logic in modelling | Hari |
| Mar 18–20 | Architecture sketches DC1 & DC2 | Harshwardhan |
| Mar 20–22 | Architecture tables (components, connectors, data elements) | Pushkar |
| Mar 22–24 | Controller integration, end-to-end testing | Hari |
| Mar 24–25 | Doc compilation, diagrams finalized | Harshwardhan |
| Mar 26 | Final fixes, cleanup, submission | All |

## Commit Message Format

```
[module] Short description

- Detail 1
- Detail 2
```

Examples:
```
[preprocess] Add data loading and deduplication functions
[model] Implement chained multi-output training logic
[docs] Add architecture sketch for Design Choice 2
```

## Git Commands for Authoring

To commit as a specific teammate:

```bash
GIT_AUTHOR_NAME="Pushkar Upadhyay" GIT_AUTHOR_EMAIL="newpusupa023@gmail.com" \
GIT_COMMITTER_NAME="Pushkar Upadhyay" GIT_COMMITTER_EMAIL="newpusupa023@gmail.com" \
git commit -m "message"
```

```bash
GIT_AUTHOR_NAME="Harshwardhan Dongare" GIT_AUTHOR_EMAIL="harshwardhan.dongare@gmail.com" \
GIT_COMMITTER_NAME="Harshwardhan Dongare" GIT_COMMITTER_EMAIL="harshwardhan.dongare@gmail.com" \
git commit -m "message"
```

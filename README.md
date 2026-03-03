# Email Classifier - EEAI CA1

Multi-label email classification system using modular software architecture.

## Team Members
- Hari Krishnan
- Pushkar Upadhyay
- Harshwardhan Dongare

## Architecture
- **main.py** - Controller module
- **preprocess.py** - Data loading and preprocessing
- **embeddings.py** - TF-IDF text vectorization
- **modelling/** - Data model and model orchestration
- **model/** - ML model implementations (RandomForest)

## Design Choices
1. **Chained Multi-Output Classification** - Single model instance classifying progressively combined labels
2. **Hierarchical Modelling** - Tree of model instances filtering data per class at each level

## How to Run
```bash
python main.py
```

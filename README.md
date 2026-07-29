# SpamShield AI

SpamShield AI is a modular Streamlit application for spam detection built from the original notebook workflow. It supports multiple text feature extraction and model combinations, exposes a prediction workspace, and includes pages for model comparison, exploratory analysis, and project documentation.

## Features

- Single-message spam prediction.
- Batch CSV scoring.
- Multiple modeling pipelines:
  - Bag of Words + Naive Bayes
  - TF-IDF + Logistic Regression
  - Word2Vec + Random Forest
  - Average Word2Vec + XGBoost
- Model comparison dashboard.
- Exploratory data analysis views.
- Project overview and roadmap pages.

## Project Structure

```text
.
├── app.py
├── pages/
├── src/
├── assets/
├── data/
├── models/
├── vectorizers/
└── notebooks/
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Data and Artifacts

- Put the dataset in the `data/` directory using the path configured in `src/config.py`.
- Saved models should live in `models/`.
- Saved vectorizers should live in `vectorizers/`.
- If artifacts are missing, the UI should fail gracefully or fall back to sample data where possible.

## Training Workflow

The runtime code in `src/` is organized to support a separate training step:

- `src/preprocessing.py` cleans and tokenizes text.
- `src/feature_extraction.py` builds or loads vectorizers and embeddings.
- `src/model.py` defines model creation and persistence.
- `src/train.py` handles dataset loading, splitting, training, and artifact saving.
- `src/evaluation.py` computes metrics and comparison outputs.
- `src/predict.py` performs runtime inference.

## Streamlit Pages

- `1_Home.py` - landing page and project overview.
- `2_Predict.py` - prediction workspace.
- `3_Model_Comparison.py` - performance comparison dashboard.
- `4_EDA.py` - exploratory analysis.
- `5_About.py` - project documentation.
- `6_Future_Work.py` - roadmap and next steps.

## Deployment

This project is intended to deploy to Streamlit Community Cloud.

Recommended deployment checklist:

1. Commit the application code, `requirements.txt`, and any required assets.
2. Confirm the app can start from `app.py`.
3. Verify the dataset and trained artifacts are available or that the app can gracefully fall back.
4. Connect the repository to Streamlit Community Cloud and select `app.py` as the entry point.

## Notes

- The codebase is intentionally modular so the notebook logic can evolve without rewriting the UI.
- The comparison and EDA pages are designed to become data-driven once persisted training outputs are wired in.
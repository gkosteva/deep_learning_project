# Fake News Detection on LIAR (Разпознаване на фалшиви новини)

Classify a short political **statement** by its truthfulness, using the
[LIAR dataset](https://github.com/tfs4/liar_dataset), and expose it through a
**Streamlit** web app. The project follows the course modelling story - a
majority-class **baseline**, a family of **embedding** and **recurrent** models,
optional **transformer** fine-tuning, a coloured Excel **Model Report File**,
and full BDD unit-test coverage.

Two tasks are supported:

- **six-class** - the full truth scale `pants-fire … true`;
- **binary** - a fake/real collapse.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the whole pipeline (baseline -> embeddings -> LSTM/GRU -> report + figures).
python run.py --task six       # 6-way truth scale (default)
python run.py --task binary    # fake/real collapse
python run.py --task six --metadata        # append speaker/party metadata to the text
python run.py --task six --transformers    # also fine-tune BERT/RoBERTa/DistilBERT/GPT-2 (slow)
```

Results are written to `reports/model_report_<task>.xlsx`, figures to
`reports/figures/<task>/`, and the best model to
`reports/artifacts/best_model_<task>.pt`. The LIAR splits are read from
`data/raw/liar/`.

The LIAR splits live in `data/raw/liar/` (`train.tsv`, `valid.tsv`,
`test.tsv`). To use real GloVe vectors, drop `glove.6B.100d.txt` under
`data/raw/`; Word2Vec and FastText are trained on the corpus automatically.

## Web app

```bash
streamlit run app/streamlit_app.py
```

Type or pick a statement, choose the task, and get the predicted truthfulness
label, a plain-language explanation, the confidence and a probability chart. The
app loads the saved best model, or falls back to a TF-IDF classifier.

## Data exploration

```bash
python experiments/exp_00_explore.py   # prints the analysis, writes reports/figures/eda/
```

## Running the experiments individually

```bash
python experiments/exp_01_baseline.py      # resets the ledger, writes row 1
python experiments/exp_02_rnn_variants.py  # LSTM / GRU / BiLSTM / BiGRU (step 4)
python experiments/exp_03_embeddings.py    # TF-IDF / Word2Vec / GloVe / FastText (step 3)
python experiments/exp_04_transformers.py  # BERT / RoBERTa / DistilBERT / GPT-2 (step 5)
```

Each script appends a row to `reports/records.jsonl` and rebuilds
`reports/model_report.xlsx`.

## Tests

```bash
coverage run -m pytest
coverage report -m
```

Behaviour-driven naming (`test_when_<condition>_then_<expectation>`), **100%**
coverage of the `src` package. Lint/format: `yapf -i -r src`, `isort src`,
`mypy src`.

## Layout

```
src/fake_news/        library code (data, models, training, reporting, inference)
app/streamlit_app.py  Streamlit web UI
experiments/          runnable scripts (exploration + experiments)
tests/                BDD unit tests
reports/              generated Excel report, figures and model artifacts
docs/                 full documentation + presentation outline
```

See [docs/documentation.md](docs/documentation.md) for the full write-up and
[docs/presentation_outline.md](docs/presentation_outline.md) for the slides.

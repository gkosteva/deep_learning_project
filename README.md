# Fake News Detection (Разпознаване на фалшиви новини)

Binary text classification that decides whether a news article is **fake** (`0`)
or **real** (`1`). The project follows the course modelling story: a
majority-class **baseline**, an LSTM **main model**, two simple **improvements**,
a coloured Excel **Model Report File**, and full BDD unit-test coverage.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the whole pipeline (baseline -> LSTM -> improvements -> report + figures).
python run.py                  # auto: real data if present, else synthetic
python run.py --data real      # ISOT real data only (errors if CSVs are missing)
python run.py --data synthetic # generated synthetic corpus only
```

`--data` selects the corpus and keeps the two runs separate: results are written
to `reports/model_report_<source>.xlsx` (e.g. `model_report_isot.xlsx` or
`model_report_synthetic.xlsx`) with figures under `reports/figures/<source>/`,
and the data source is recorded in the report title and a `data` column. The
default `auto` falls back to a small synthetic corpus when the ISOT files are
absent so it runs out of the box.

To use the real data, download the
[ISOT Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
and place `Fake.csv` and `True.csv` under `data/raw/`.

## Running the experiments individually

```bash
python experiments/exp_01_baseline.py       # resets the ledger, writes row 1
python experiments/exp_02_lstm.py           # main model
python experiments/exp_03_lstm_dropout.py   # improvement 1: dropout
python experiments/exp_04_bilstm.py         # improvement 2: bidirectional
python experiments/exp_06_bilstm_large.py   # improvement 3: wider & deeper
python experiments/exp_07_bilstm_adamw.py   # improvement 4: AdamW weight decay
python experiments/exp_08_glove.py          # improvement 5: GloVe embeddings
python experiments/exp_05_transformer.py    # optional DistilBERT stretch (run last)
```

To use real GloVe vectors, download
[`glove.6B.zip`](https://nlp.stanford.edu/data/glove.6B.zip) and place
`glove.6B.100d.txt` under `data/raw/`. Without it the GloVe experiment falls back
to deterministic stand-in vectors so it still runs.

Each script appends a row to `reports/records.jsonl` and rebuilds
`reports/model_report.xlsx`.

## Tests

```bash
coverage run -m pytest
coverage report -m
```

The suite uses behaviour-driven naming (`test_when_<condition>_then_<expectation>`)
and reaches 100% coverage of the `src` package.

## Layout

```
src/fake_news/        library code (data, models, training, reporting)
experiments/          one runnable script per experiment
tests/                BDD unit tests
reports/              generated Excel report and figures
docs/documentation.md full project documentation
```

See [docs/documentation.md](docs/documentation.md) for the full write-up.

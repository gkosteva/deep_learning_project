# Presentation outline - Fake News Detection on LIAR

A suggested slide deck describing the approach (step 7). One bullet block per
slide.

## Slide 1 - Title
- Разпознаване на фалшиви новини / Fake News Detection
- Classify a statement by its truthfulness - LIAR dataset + web app
- Author, course, date

## Slide 2 - Problem & motivation
- Misinformation spreads faster than fact-checking
- Goal: given a short statement, predict how truthful it is
- Two framings: 6-way truth scale and binary fake/real

## Slide 3 - Related work / techniques (step 1)
- Wang 2017, "Liar, Liar Pants on Fire" (LIAR benchmark)
- Survey of fake-news detection: linguistic, neural, transformer approaches
- Embeddings: TF-IDF → Word2Vec/GloVe/FastText → contextual (BERT family)

## Slide 4 - The data (step 2)
- 12,836 statements, official train/valid/test split
- 14 features: statement + speaker/party/subject/credit-history metadata
- Label distribution (balanced 6-way), statement length ≈ 18 words
- Anomalies: 129 missing contexts, 26 duplicates
- Show EDA figures (label distribution, lengths, credit-history correlation)

## Slide 5 - Why LIAR is hard
- One short sentence, subjective adjacent labels, no external evidence
- Literature: ~0.27 six-class / ~0.65 binary accuracy

## Slide 6 - Pipeline & preprocessing
- clean_text → vocabulary → padded sequences
- Baseline (majority class) as the reference point

## Slide 7 - Embedding experiments (step 3)
- TF-IDF + logistic regression
- Learned Embedding layer
- Word2Vec / FastText trained on the corpus, GloVe loaded/stand-in
- Take-away: which embedding helped most

## Slide 8 - Recurrent models (step 4)
- LSTM vs GRU, uni- vs bidirectional, dropout, depth/width, AdamW
- Masked mean pooling diagram

## Slide 9 - Transformer models (step 5)
- BERT, RoBERTa, DistilBERT, GPT-2 fine-tuning
- Trade-off: accuracy vs cost; opt-in in the pipeline

## Slide 10 - Results (Model Report File)
- Excel report walkthrough: metrics + % change vs baseline, best model
- Confusion matrix and train/val curves of the best model

## Slide 11 - Web application (step 6)
- Streamlit demo: type a statement → verdict + probabilities
- Live demo / screenshot

## Slide 12 - Testing & engineering
- BDD unit tests, 100% coverage; yapf/isort/mypy clean
- Reproducible (fixed seeds); GloVe/TF-IDF fallbacks keep it runnable offline

## Slide 13 - Limitations & future work
- English only, no external evidence, subjective scale
- Metadata fusion, evidence retrieval, ordinal loss, calibration/explainability

## Slide 14 - Q&A
```

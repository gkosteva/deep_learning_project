from typing import Dict, List, Sequence, Tuple

# Human-readable explanation for every label the UI can show.
TRUTH_DESCRIPTIONS: Dict[str, str] = {
    'pants-fire': 'Pants on fire - a ridiculous, blatantly false claim.',
    'false': 'False - the statement is not accurate.',
    'barely-true': 'Barely true - mostly inaccurate, with a grain of truth.',
    'half-true': 'Half true - partially accurate but missing context.',
    'mostly-true': 'Mostly true - accurate with minor caveats.',
    'true': 'True - the statement is accurate.',
    'fake': 'Fake - the statement is likely misinformation.',
    'real': 'Real - the statement is likely trustworthy.',
}


def describe(label: str) -> str:
    return TRUTH_DESCRIPTIONS.get(label, label)


def top_prediction(ranked: Sequence[Tuple[str, float]]) -> Tuple[str, float]:
    if not ranked:
        raise ValueError('ranked predictions must not be empty')
    return ranked[0]


def format_probabilities(ranked: Sequence[Tuple[str, float]]) -> List[Dict[str, object]]:
    return [{
        'label': label,
        'probability': float(probability),
        'percent': round(float(probability) * 100, 1),
    } for label, probability in ranked]


class TfidfService:
    """Wraps a fitted TF-IDF classifier behind the PredictionService interface."""

    def __init__(self,
                 model,
                 labels: Sequence[str],
                 model_name: str = 'TF-IDF + Logistic Regression (fallback)'):
        self.model = model
        self.labels = list(labels)
        self.model_name = model_name

    def predict_proba(self, statement: str) -> List[Tuple[str, float]]:
        probabilities = self.model.predict_proba([statement])[0]
        ranked = sorted(zip(self.labels, probabilities), key=lambda item: item[1], reverse=True)
        return [(label, float(probability)) for label, probability in ranked]

    def predict(self, statement: str) -> str:
        return self.predict_proba(statement)[0][0]

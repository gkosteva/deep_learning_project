from typing import List, Sequence


class TfidfLogisticClassifier:
    """TF-IDF features fed to a multinomial logistic-regression classifier."""

    def __init__(self,
                 max_features: int = 20000,
                 ngram_max: int = 2,
                 c: float = 4.0,
                 seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            sublinear_tf=True,
            min_df=2,
        )
        self.classifier = LogisticRegression(max_iter=2000, C=c, random_state=seed)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> 'TfidfLogisticClassifier':
        features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(features, labels)
        return self

    def predict(self, texts: Sequence[str]) -> List[int]:
        features = self.vectorizer.transform(texts)
        return self.classifier.predict(features).tolist()

    def predict_proba(self, texts: Sequence[str]):
        features = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(features)

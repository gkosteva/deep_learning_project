import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fake_news.app_support import TfidfService, describe, format_probabilities, top_prediction
from src.fake_news.config import class_names
from src.fake_news.data.dataset import label_column, load_liar, select_text
from src.fake_news.inference import PredictionService, artifact_path
from src.fake_news.models.tfidf_classifier import TfidfLogisticClassifier

st.set_page_config(page_title='LIAR Fake-News Classifier', page_icon='🔍', layout='centered')

EXAMPLES = [
    'The unemployment rate has dropped to the lowest level in fifty years.',
    'Our state spends more on prisons than on public schools.',
    'This policy will create two million new jobs in a single year.',
]


@st.cache_resource(show_spinner=False)
def load_service(task: str):
    path = artifact_path(task)
    if os.path.exists(path):
        return PredictionService.load(path)
    return _train_fallback(task)


def _train_fallback(task: str) -> TfidfService:
    train_df, _, _ = load_liar('data/raw/liar')
    column = label_column(task)
    model = TfidfLogisticClassifier().fit(select_text(train_df, False), train_df[column].tolist())
    return TfidfService(model, class_names(task))


def main() -> None:
    st.title('🔍 LIAR Fake-News Classifier')
    st.caption('Classify a political statement by its truthfulness, trained on the LIAR dataset.')

    with st.sidebar:
        st.header('Settings')
        task = st.radio('Classification task', ['six', 'binary'],
                        format_func=lambda t: '6-way truth scale'
                        if t == 'six' else 'Binary fake/real')
        st.markdown('---')
        st.markdown('Run `python run.py --task six` to train and save the best model. '
                    'Without a saved model the app falls back to a TF-IDF classifier.')

    service = load_service(task)
    st.info(f'Active model: **{service.model_name}**')

    st.subheader('Statement')
    example = st.selectbox('Try an example (optional)', [''] + EXAMPLES)
    statement = st.text_area('Enter a statement to classify', value=example, height=120)

    if st.button('Classify', type='primary'):
        if not statement.strip():
            st.warning('Please enter a statement first.')
            return
        ranked = service.predict_proba(statement)
        label, confidence = top_prediction(ranked)
        st.markdown(f'### Verdict: `{label}`')
        st.write(describe(label))
        st.metric('Confidence', f'{confidence * 100:.1f}%')

        rows = format_probabilities(ranked)
        chart_data = pd.DataFrame(rows).set_index('label')[['probability']]
        st.bar_chart(chart_data)
        st.dataframe(pd.DataFrame(rows)[['label', 'percent']],
                     hide_index=True,
                     use_container_width=True)


if __name__ == '__main__':
    main()

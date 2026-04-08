import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "nsdemidov/tuned-deberta-v3-base-for-arxiv-classification"
MAX_LENGTH = 128
DISPLAY_CUMSUM_THRESHOLD = 0.95 

EXAMPLE_TITLE = "Attention Is All You Need"
EXAMPLE_ABSTRACT = (
    "The dominant sequence transduction models are based on complex recurrent or convolutional "
    "neural networks that include an encoder and a decoder. We propose a new simple network "
    "architecture, the Transformer, based solely on attention mechanisms, dispensing with "
    "recurrence and convolutions entirely."
)

st.set_page_config(
    page_title="Распознование статьи на Arxiv",
    page_icon="📄",
    layout="centered"
)

@st.cache_resource(show_spinner="Загрузка модели...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32
    )

    device = torch.device("cpu")

    model.to(device)

    if device.type == "cpu":
        model = model.float()

    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

def keep_top_until_threshold(results, threshold=0.95):
    filtered = []
    cumulative = 0.0

    for label, score in results:
        filtered.append((label, score))
        cumulative += score

        if cumulative >= threshold:
            break

    return filtered

def predict(title: str, abstract: str | None) -> dict:
    title = title.strip()
    abstract = (abstract or "").strip()

    if abstract:
        text = title + " [SEP] " + abstract
    else:
        text = title

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    sorted_ids = torch.argsort(probs, descending=True).tolist()

    all_results = []
    for idx in sorted_ids:
        label = model.config.id2label.get(idx, str(idx))
        score = probs[idx].item()
        all_results.append((label, score))

    visible_results = keep_top_until_threshold(all_results, DISPLAY_CUMSUM_THRESHOLD)

    return {
        "label": all_results[0][0],
        "score": all_results[0][1],
        "all": all_results,
        "visible": visible_results
    }

st.title("Классификация научных статей по тематикам из хранилища ArXiv")

with st.container(border=True):
    st.subheader("О проекте")
    st.markdown(
        """
        Это приложение определяет наиболее вероятную тематику научной статьи
        по её **заголовку** и **abstract**.
        Для обучения я дообучал Bert-like модель(а именно deberta-v3-base) .

        **Как использовать это приложение:**
        1. Введите заголовок статьи.
        2. Введите abstract статьи (необязательно).
        3. Нажмите **«Классифицировать»**.
        4. Приложение покажет наиболее вероятный жанр и распределение вероятностей.

        **Важно:** если указать abstract, результат обычно получается точнее (так как изначально модель
        изначально училась предсказывать жанр по заголовку и abstract).
        """
    )

    with st.expander("Пример входных данных"):
        st.markdown("**Заголовок:**")
        st.code(EXAMPLE_TITLE, language=None)

        st.markdown("**Abstract:**")
        st.code(EXAMPLE_ABSTRACT, language=None)

st.divider()

title = st.text_input(
    "Заголовок статьи *",
    placeholder="Attention Is All You Need"
)

abstract = st.text_area(
    "Abstract",
    placeholder="(необязательно)",
    height=150
)

button_classification = st.button(
    "Классифицировать",
    type="primary",
    use_container_width=True
)

if button_classification:
    if not title.strip():
        st.warning("Заголовок обязателен (смотри пример)")

    else:
        with st.spinner("Анализируем.."):
            result = predict(title, abstract)

        st.divider()
        st.subheader("Результат")

        col1, col2 = st.columns(2)
        col1.metric("Жанр", result["label"])
        col2.metric("Уверенность", f"{result['score']:.1%}")

        with st.expander("Наиболее вероятные тематики"):
            st.caption("тематики, которые суммарно покрывают 95% вероятности.")
            for label, score in result["visible"]:
                st.progress(score, text=f"{label}: {score:.1%}")

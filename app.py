import streamlit as st

from predict import MODEL_PATH, load_model, predict_pair


st.set_page_config(page_title="Duplicate Question Pairs", page_icon="💬")
st.title("Duplicate Question Pairs")
st.write("Do these questions ask the same thing?")

if not MODEL_PATH.exists():
    st.info("Train the model first: python train.py")
    st.stop()


@st.cache_resource
def get_model(modified_time):
    return load_model()


artifacts = get_model(MODEL_PATH.stat().st_mtime_ns)

with st.form("questions"):
    q1 = st.text_area("First question", max_chars=2000)
    q2 = st.text_area("Second question", max_chars=2000)
    submitted = st.form_submit_button("Compare questions")

if submitted:
    try:
        label, score = predict_pair(q1, q2, artifacts)
        if label:
            st.success("Likely duplicate questions")
        else:
            st.info("Likely different questions")
        st.metric("Duplicate score", f"{score:.1%}")
        st.caption("This model score is not a calibrated certainty. Review important matches yourself.")
    except ValueError as error:
        st.warning(str(error))

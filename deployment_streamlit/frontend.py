import streamlit as st
from utils import send_request
from classes import GogohiGPTRequest

st.set_page_config(page_title="GogohiGPT", page_icon="🤘")

st.title("GogohAI Lab X - Beta")
st.header("Welcome to GogohiGPT.")


def app():
    def callback():
        st.session_state.submit_text = True
    if "submit_text" not in st.session_state:
        st.session_state.submit_text = False
    st.sidebar.title("Hyperparameters")
    max_new_tokens = st.sidebar.slider("Maximum Number of New Tokens", min_value=1, max_value=500, value=50)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.9)
    top_p = st.sidebar.slider("Top P", min_value=0.01, max_value=1.0, value=0.85)
    top_k = st.sidebar.slider("Top K", min_value=0, max_value=100, value=40)
    do_sample = st.sidebar.radio("Nucleus Sampling", options=[True, False])

    with st.form(key="GogohiGPT Prediction"):
        prompt = st.text_area("Prompt")
        st.session_state.submit_text = st.form_submit_button("Submit", on_click=callback) or st.session_state.submit_text

    if st.session_state.submit_text:
        st.success("GogohiGPT Prediction")
        request = GogohiGPTRequest(prompt=prompt,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature,
                                   top_p=top_p,
                                   top_k=top_k,
                                   do_sample=do_sample)
        response = send_request(request)
        st.write(response.json()["text"])


if __name__ == "__main__":
    app()

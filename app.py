import streamlit as st
import requests

st.title("📚 AI Research Assistant")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    res = requests.post("http://127.0.0.1:8000/upload", files=files)
    st.success("PDF uploaded and processed!")

# Ask Question
question = st.text_input("Ask a question")

if st.button("Get Answer"):
    res = requests.post(
        "http://127.0.0.1:8000/query",
        json={"question": question}
    )
    answer = res.json()["answer"]
    st.write("### Answer:")
    st.write(answer)
import streamlit as st
import pdfplumber 
from gensim.summarization import summarize

def extract_text(feed):
    text = ""
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

st.title("PDF Text Extractor and Summarizer")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
    extracted_text = extract_text(uploaded_file)
    if extracted_text:
        st.subheader("Extracted Text:")
        st.text(extracted_text)
        
        st.subheader("Summarized Text:")
        summarized_text = summarize(extracted_text, ratio=0.25)
        st.text(summarized_text)
    else:
        st.write("No text found in the PDF file.")

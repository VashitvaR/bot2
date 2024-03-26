import streamlit as st
import pdfplumber 

def extract_text(feed):
    text = ""
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

st.title("PDF Text Extractor")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
    extracted_text = extract_text(uploaded_file)
    if extracted_text:
        st.text(extracted_text)
    else:
        st.write("No text found in the PDF file.")

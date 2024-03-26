import streamlit as st
import pdfplumber 
import pandas as pd

def extract_data(feed):
    tables_data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for page in pages:
            tables = page.extract_tables()
            for table in tables:
                tables_data.append(table)
    # Convert list of tables into a DataFrame
    if tables_data:
        df = pd.DataFrame()
        for table_data in tables_data:
            df = pd.concat([df, pd.DataFrame(table_data)], ignore_index=True)
        return df
    else:
        return None

st.title("PDF Table Extractor")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
    extracted_df = extract_data(uploaded_file)
    if extracted_df is not None:
        st.dataframe(extracted_df)
    else:
        st.write("No tables found in the PDF file.")

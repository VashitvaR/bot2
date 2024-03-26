import streamlit as st
import PyPDF2

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

def main():
    st.title("PDF Text Extractor")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Display the uploaded PDF
        st.write("### Uploaded PDF:")
        st.write(uploaded_file)

        # Extract text from the PDF
        text = extract_text_from_pdf(uploaded_file)
        
        # Display extracted text
        st.write("### Extracted Text:")
        st.write(text)

if __name__ == "__main__":
    main()

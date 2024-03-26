import streamlit as st

# Set page title
st.title("PDF File Uploader")

# Display file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Check if a file was uploaded
if uploaded_file is not None:
    # Read and display file content
    file_contents = uploaded_file.read()
    st.write("File content:")
    st.write(file_contents)

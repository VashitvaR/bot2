import streamlit as st
import pdfplumber 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq
nltk.download('punkt')
nltk.download('stopwords')
def extract_text(feed):
    text = ""
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def summarize_text(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    
    # Calculate word frequencies
    word_frequencies = FreqDist(word for word in text.split() if word.lower() not in stop_words)
    
    # Calculate sentence scores based on word frequencies
    sentence_scores = {sentence: sum(word_frequencies[word] for word in sentence.split()) for sentence in sentences}
    
    # Get top N sentences with the highest scores
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    return ' '.join(summary_sentences)

st.title("PDF Text Extractor and Summarizer")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
    extracted_text = extract_text(uploaded_file)
   
    st.subheader("Summarized Text:")
    summarized_text = summarize_text(extracted_text)
    st.text(summarized_text)
else:
    st.write("No text found in the PDF file.")

import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the BERT model and tokenizer
with st.sidebar:
    st.markdown('<h1 style="color:blue;">🛡️ Welcome to Shield Bot by Nayak! 🤖</h1>', unsafe_allow_html=True)
    st.header('About Nayak')

    st.markdown('''
    ## Empowering Reporting and Insights

    Nayak is a cutting-edge reporting and insights platform designed to empower individuals, particularly victims, to file reports openly or anonymously. The platform is committed to user convenience and offers morning and night modes.

    🚀 Once a report is submitted, Nayak provides authorized authorities with a comprehensive dashboard. Reports are intelligently categorized into open, in-progress, and closed statuses, ensuring efficient case management.

    ### AI-Powered Efficiency
    - Nayak seamlessly integrates AI to manage the influx of reports effectively.
    - The AI is adept at generating precise answers to specific questions posed by authorities, streamlining the information processing workflow.

    ### API Gateway for Data Insights
    - Nayak offers an API gateway, allowing authorities to access valuable data stored within the platform's databases.
    - Authorized external parties can retrieve pertinent information through the API, contributing to insights generation.
    

    💡 Note: The  Bot is an integral component of Nayak, augmenting the platform's capabilities and enhancing user experience.
    ''')
st.markdown('<style>div.stNamedPlaceholder>div{margin-top:20px;}</style>', unsafe_allow_html=True)
tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to get model output
def get_model_output(user_input):
    max_length = 512
    # Tokenize the user input
    input_tokens = tokenizer.tokenize(user_input)

    # Pad the input tokens
    input_tokens = input_tokens + [tokenizer.pad_token] * (max_length - len(input_tokens))

    # Convert the input tokens to input ids
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    # Create the attention mask for the input
    attention_mask = [1 if token != tokenizer.pad_token else 0 for token in input_tokens]

    # Convert the input ids and attention mask to tensors
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    # Get the model output
    output = model(input_ids, attention_mask=attention_mask)

    # Get the predicted label
    prediction = output[0].argmax(dim=1).item()

    return prediction

# Streamlit app
st.title("Nayak QA System")

# Take input from the user
user_input = st.text_input("Enter your text:")

# Check if the user has entered any text
if user_input:
    # Get the model output
    prediction = get_model_output(user_input)

    # Print the output
    if prediction == 0:
        st.write("Question: {}".format(user_input))
    else:
        st.write("Answer: {}".format(answers[prediction - 1]))

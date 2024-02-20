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
data = [
    ("How is Nayak always looking to improve and enhance its features?", "Nayak is always looking for ways to improve and enhance its features. If you have any suggestions or specific areas you'd like to see improvements, feel free to share them with us!"),
    ("What is Nayak's approach to addressing societal challenges?", "Nayak integrates artificial intelligence to contribute to its effectiveness in addressing societal challenges. The goal is to provide a platform that is continually evolving and making a positive impact."),
    ("Does Nayak have specific success stories?", "While Nayak doesn't have specific success stories, its focus is on creating positive outcomes for individuals facing societal challenges. Your feedback and experiences contribute to the ongoing improvement of the platform."),
    ("What are Nayak's goals in addressing abuse, fraud, and danger?", "Nayak aims to influence policy changes and raise awareness about abuse, fraud, and danger. By connecting victims with authorities, the platform seeks to address conflicts and empower those affected."),
    ("How does Nayak address challenges or barriers?", "Nayak acknowledges there may be challenges or barriers. If you have faced any difficulties using the platform, we'd like to hear more about your experiences to make necessary improvements."),
    ("How can individuals contribute to building a supportive community using Nayak?", "If you've found success using Nayak, recommending the platform to others facing similar challenges can help build a supportive community and amplify its impact."),
    ("How does Nayak encourage collaboration with support systems and authorities?", "Nayak encourages collaboration with existing support systems and authorities to create a comprehensive solution. Together, we can work towards a safer and more supportive environment."),
    ("What role does Nayak play in supporting individuals, communities, and governments?", "Nayak plays a crucial role in supporting individuals, communities, and governments. By promoting collaboration and providing valuable insights, the platform aims to make a positive impact in addressing societal challenges."),
    ("How does Nayak aim to foster a sense of community and solidarity?", "Nayak strives to foster a sense of community and solidarity among individuals facing societal challenges. Your participation and engagement contribute to creating a supportive environment."),
    ("In what way is Nayak open to feedback for continuous improvement?", "Nayak is open to feedback on how it can better address specific challenges and empower victims. Your insights and suggestions play a vital role in the continuous improvement of the platform.")
]



questions = [item[0] for item in data]
answers = [item[1] for item in data]
# Convert the data to input format for BERT
question_input_ids = []
question_attention_masks = []
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

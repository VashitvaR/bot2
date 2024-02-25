import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances

# Nayak-related data
data=[
  {
    "question": "How is Nayak always looking to improve and enhance its features?",
    "answer": "Nayak is always looking for ways to improve and enhance its features. If you have any suggestions or specific areas you'd like to see improvements, feel free to share them with us!"
  },
  {
    "question": "What is Nayak's approach to addressing societal challenges?",
    "answer": "Nayak integrates artificial intelligence to contribute to its effectiveness in addressing societal challenges. The goal is to provide a platform that is continually evolving and making a positive impact."
  },
  {
    "question": "Does Nayak have specific success stories?",
    "answer": "While Nayak doesn't have specific success stories, its focus is on creating positive outcomes for individuals facing societal challenges. Your feedback and experiences contribute to the ongoing improvement of the platform."
  },
  {
    "question": "What are Nayak's goals in addressing abuse, fraud, and danger?",
    "answer": "Nayak aims to influence policy changes and raise awareness about abuse, fraud, and danger. By connecting victims with authorities, the platform seeks to address conflicts and empower those affected."
  },
  {
    "question": "How does Nayak address challenges or barriers?",
    "answer": "Nayak acknowledges there may be challenges or barriers. If you have faced any difficulties using the platform, we'd like to hear more about your experiences to make necessary improvements."
  },
  {
    "question": "How can individuals contribute to building a supportive community using Nayak?",
    "answer": "If you've found success using Nayak, recommending the platform to others facing similar challenges can help build a supportive community and amplify its impact."
  },
  {
    "question": "How does Nayak encourage collaboration with support systems and authorities?",
    "answer": "Nayak encourages collaboration with existing support systems and authorities to create a comprehensive solution. Together, we can work towards a safer and more supportive environment."
  },
  {
    "question": "What role does Nayak play in supporting individuals, communities, and governments?",
    "answer": "Nayak plays a crucial role in supporting individuals, communities, and governments. By promoting collaboration and providing valuable insights, the platform aims to make a positive impact in addressing societal challenges."
  },
  {
    "question": "How does Nayak aim to foster a sense of community and solidarity?",
    "answer": "Nayak strives to foster a sense of community and solidarity among individuals facing societal challenges. Your participation and engagement contribute to creating a supportive environment."
  },
  {
    "question": "In what way is Nayak open to feedback for continuous improvement?",
    "answer": "Nayak is open to feedback on how it can better address specific challenges and empower victims. Your insights and suggestions play a vital role in the continuous improvement of the platform."
  },

  
  {
    "question": "How does Nayak ensure user privacy and data security?",
    "answer": "User privacy and data security are top priorities for Nayak. Stringent measures are in place to safeguard your information, and continuous improvements are made to meet evolving security standards."
  },
  {
    "question": "What steps does Nayak take to combat misinformation on its platform?",
    "answer": "Nayak employs advanced algorithms and fact-checking mechanisms to combat misinformation. User reporting is also crucial in identifying and addressing false information promptly."
  },
  {
    "question": "How does Nayak collaborate with local communities to tailor its services?",
    "answer": "Nayak actively collaborates with local communities to tailor its services. Your input and insights help shape the platform to better meet the unique needs of diverse communities."
  },
  {
    "question": "What measures does Nayak have in place to promote inclusivity and diversity?",
    "answer": "Inclusivity and diversity are core values for Nayak. The platform is designed to be inclusive, and efforts are ongoing to ensure diverse voices are represented and heard."
  },
  {
    "question": "Can Nayak be accessed by individuals with disabilities, and what accessibility features are available?",
    "answer": "Nayak is committed to accessibility. Features such as screen reader compatibility and keyboard shortcuts are implemented to ensure individuals with disabilities can access and use the platform."
  },
  {
    "question": "How does Nayak handle and respond to reports of inappropriate content?",
    "answer": "Nayak takes reports of inappropriate content seriously. A dedicated team reviews reports promptly, and appropriate action is taken to maintain a safe and respectful environment for all users."
  },
  {
    "question": "What role does Nayak envision in the future of addressing global challenges?",
    "answer": "Nayak envisions a significant role in addressing global challenges. Continued advancements in technology and collaboration with international partners aim to contribute to positive global outcomes."
  },
  {
    "question": "How does Nayak adapt to evolving user needs and preferences?",
    "answer": "Adaptability is a key focus for Nayak. Regular feedback surveys and user interactions help the platform adapt to evolving user needs and preferences, ensuring a user-friendly experience."
  },
  {
    "question": "What resources does Nayak provide for users dealing with mental health challenges?",
    "answer": "Nayak recognizes the importance of mental health. Resources, support networks, and information on mental health services are made available to assist users dealing with mental health challenges."
  },
  {
    "question": "How can users actively participate in shaping Nayak's future developments?",
    "answer": "Users play a vital role in shaping Nayak's future developments. Participation in beta testing, providing feedback, and engaging in community discussions contribute to the platform's ongoing improvement."
  },
  {
    "question": "What measures does Nayak take to prevent and address online harassment?",
    "answer": "Online harassment prevention is a priority for Nayak. Robust reporting mechanisms, AI algorithms, and cooperation with law enforcement help identify and address instances of online harassment."
  },
  {
    "question": "How does Nayak handle situations where user safety is compromised?",
    "answer": "User safety is paramount for Nayak. Emergency response features, quick access to support services, and a dedicated safety team contribute to handling situations where user safety is compromised."
  },
  {
    "question": "What steps does Nayak take to ensure transparency in its decision-making processes?",
    "answer": "Nayak is committed to transparency. Clear communication, regular updates, and transparent decision-making processes are implemented to keep users informed and involved in the platform's developments."
  },
  {
    "question": "How does Nayak contribute to educational initiatives and awareness campaigns?",
    "answer": "Nayak actively contributes to educational initiatives and awareness campaigns. Partnerships with educational institutions and the creation of informative content aim to raise awareness on various societal issues."
  },
  {
    "question": "How can Nayak users connect with each other for mutual support and encouragement?",
    "answer": "Nayak provides features for users to connect with each other. Community forums, support groups, and networking opportunities are available to facilitate mutual support and encouragement among users."
  },
  {
    "question": "What role does Nayak see in promoting digital literacy and online safety?",
    "answer": "Nayak recognizes the importance of digital literacy and online safety. Initiatives, resources, and collaborations with educational institutions aim to promote responsible digital citizenship and online safety."
  },
  {
    "question": "How does Nayak ensure fair and unbiased algorithmic decision-making?",
    "answer": "Fairness and unbiased algorithmic decision-making are priorities for Nayak. Regular audits, diversity in development teams, and ongoing refinement of algorithms contribute to minimizing biases in the platform."
  },
  {
    "question": "What support does Nayak offer to individuals transitioning from crisis situations to stability?",
    "answer": "Nayak offers comprehensive support to individuals transitioning from crisis situations to stability. Access to resources, counseling services, and connections with support networks are key components of this support."
  },
  {
    "question": "How does Nayak engage with policymakers to advocate for positive societal changes?",
    "answer": "Nayak actively engages with policymakers to advocate for positive societal changes. Collaboration, data sharing, and policy discussions are integral to influence positive changes on a broader scale."
  },
  {
    "question": "What role does Nayak play in promoting environmental sustainability and awareness?",
    "answer": "Nayak recognizes the importance of environmental sustainability. Initiatives, partnerships, and awareness campaigns are undertaken to promote environmental consciousness and sustainable practices."
  },
  {
    "question": "How does Nayak handle situations where its technology faces ethical dilemmas?",
    "answer": "Ethical considerations are central to Nayak's operations. A dedicated ethics committee, ongoing ethical reviews, and user input contribute to addressing and resolving ethical dilemmas that may arise."
  },
  {
    "question": "What measures does Nayak take to prevent and address cyberbullying on its platform?",
    "answer": "Preventing and addressing cyberbullying is a priority for Nayak. Robust reporting tools, education initiatives, and cooperation with relevant authorities help combat cyberbullying on the platform."
  },
  {
    "question": "How does Nayak collaborate with mental health professionals to enhance its support services?",
    "answer": "Nayak collaborates with mental health professionals to enhance its support services. Ongoing partnerships, consultations, and input from experts contribute to the development of effective mental health support resources."
  },
  {
    "question": "What role does Nayak see in supporting marginalized communities and promoting equity?",
    "answer": "Supporting marginalized communities and promoting equity are core values for Nayak. Initiatives, partnerships, and resource allocation are focused on addressing the unique challenges faced by marginalized communities."
  },
  
  {
    "question": "How can Nayak users report potential security vulnerabilities and contribute to platform safety?",
    "answer": "Nayak encourages users to report potential security vulnerabilities. A dedicated security team reviews reports, and a responsible disclosure program is in place to ensure the prompt resolution of identified vulnerabilities"
  }
]
additional_data = [
    {
        "question": "Hello",
        "answer": "Hello! How can Nayak assist you today?"
    },
    {
        "question": "Hi",
        "answer": "Hi there! How can Nayak help you?"
    },
    {
        "question": "Good morning",
        "answer": "Good morning! What can Nayak do for you today?"
    },
    {
        "question": "Goodbye",
        "answer": "Goodbye! If you have more questions, feel free to return anytime."
    },
    {
        "question": "Bye",
        "answer": "Bye! If there's anything else you'd like to know, don't hesitate to ask."
    },
]

# Combine existing data with additional data
updated_data = data + additional_data

# Update df_nayak with the new dataset
df_nayak = pd.DataFrame(updated_data)

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
    

    💡 Note: The Bot is an integral component of Nayak, augmenting the platform's capabilities and enhancing user experience.
    ''')

st.markdown('<style>div.stNamedPlaceholder>div{margin-top:20px;}</style>', unsafe_allow_html=True)

# Bag of Words (BOW)
def handle_greetings_goodbyes(user_input):
    user_input = user_input.lower()
    if user_input in ["hello", "hi", "good morning"]:
        return "Hello! How can Nayak assist you today?"
    elif user_input in ["goodbye", "bye"]:
        return "Goodbye! If you have more questions, feel free to return anytime."
    else:
        return None  # Return None if it's not a greeting or goodbye

cv = CountVectorizer()
x_bow = cv.fit_transform(df_nayak['question']).toarray()
features_bow = cv.get_feature_names()
df_bow = pd.DataFrame(x_bow, columns=features_bow)

def chat_bow(question):
    tidy_question = question.lower()
    cv_ = cv.transform([tidy_question]).toarray()
    cos = 1 - pairwise_distances(df_bow, cv_, metric='cosine')
    index_value = cos.argmax()

    return df_nayak['answer'].loc[index_value]

# TF-IDF
tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(df_nayak['question']).toarray()
features_tfidf = tfidf.get_feature_names()
df_tfidf = pd.DataFrame(x_tfidf, columns=features_tfidf)

def chat_tfidf(question):
    tidy_question = question.lower()
    tf = tfidf.transform([tidy_question]).toarray()
    cos = 1 - pairwise_distances(df_tfidf, tf, metric='cosine')
    index_value = cos.argmax()

    return df_nayak['answer'].loc[index_value]

# Streamlit App
def main():
    st.title("Nayak Chatbot")

    # User input
    user_input_question = st.text_input("Ask Nayak a question:")

    # Method selection
    method = st.radio("Select Chatbot Method:", ['Bag of Words (BOW)', 'TF-IDF'])

    # Chatbot response
    if st.button("Get Nayak's Answer"):
        if method == 'Bag of Words (BOW)':
            response = chat_bow(user_input_question)
        elif method == 'TF-IDF':
            response = chat_tfidf(user_input_question)
        else:
            response = "Select a valid method."

        # Default response if no match is found
        if response is None:
            response = "I'm sorry, I didn't understand that. Please try asking in a different way or provide more details."

        st.text_area("Nayak's Response:", response)

if __name__ == "__main__":
    main()

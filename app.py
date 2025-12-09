# Importing relevant Python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO
# Preprocessing
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
# Modeling
from sklearn import svm
# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Adding a Twitter-style logo at the top of the app
logo_url = "https://cdn-icons-png.flaticon.com/512/733/733579.png"  # Twitter logo icon URL
response = requests.get(logo_url)
logo_image = Image.open(BytesIO(response.content))

# Adding custom CSS to style the app
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    h1 {
        color: #1DA1F2;
    }
    h2 {
        color: #1DA1F2;
    }
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
    }
    .stTextInput>div>div>input {
        border-color: #1DA1F2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Creating page sections
site_header = st.container()
business_context = st.container()
data_desc = st.container()
performance = st.container()
tweet_input = st.container()
model_results = st.container()
sentiment_analysis = st.container()
contact = st.container()

# Site Header with Twitter Logo
with site_header:
    st.image(logo_image, width=80)  # Display Twitter logo
    st.title('Twitter Hate Speech & Abusive Language Detection and Moderation Classifier')
    st.write("""
    Created by [Priyangshu Chandra Das](https://github.com/5PCD3)
    
    This project aims to **automate content moderation** to identify hate speech using **machine learning binary classification algorithms.**
    
    Models included Random Forest, Naive Bayes, Logistic Regression, and Support Vector Machine (SVM). The final model was a **Logistic Regression** model that used Count Vectorization for feature engineering. It produced an F1 of 0.9616 and Recall (TPR) of 0.9363.  
    
    Check out the project repository [here](https://github.com/5PCD3/Twitter-Hate-Speech-Abusive-Language-Detection-and-Moderation-Classifier.git).
    """)

# Business Context
with business_context:
    st.header('The Problem of Content Moderation')
    st.write("""
    **Human content moderation exploits people by consistently traumatizing and underpaying them.** In 2019, an [article](https://www.theverge.com/2019/6/19/18681845/facebook-moderator-interviews-video-trauma-ptsd-cognizant-tampa) on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, **every major tech company**, including **Twitter**, uses human moderators to some extent, both domestically and overseas.
    
    Hate speech is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion, or sexual orientation.**
    """)

# Data Description
with data_desc:
    understanding, venn = st.columns(2)
    with understanding:
        st.text('')
        st.write("""
        The **data** for this project was sourced from a Cornell University [study](https://github.com/t-davidson/hate-speech-and-offensive-language) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.
        
        The `.csv` file has **24,783 rows** where almost **17% of the tweets were labeled as "Hate Speech & Abusive Language".**

        Each tweet's label was voted on by crowdsource and determined by majority rules.
        """)
    with venn:
        st.image(Image.open('plots/word_venn.png'), width=400)

# Model Performance Section
with performance:
    description, conf_matrix = st.columns(2)
    with description:
        st.header('Final Model Performance')
        st.write("""
        These scores are indicative of the two major roadblocks of the project:
        - The massive class imbalance of the dataset
        - The model's inability to identify what constitutes hate speech & abusive language
        """)
    with conf_matrix:
        st.image(Image.open('plots/Log_reg_confusion_matrix_normalized.png'), width=400)

# Tweet Input Section
with tweet_input:
    st.header('Is Your Tweet Considered Hate Speech or Does it Contain Abusive Language?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    # User input for tweet
    user_text = st.text_input('Enter Tweet', max_chars=280)
    # Debugging step to show input
    st.write("Debug: User input received: ", user_text)

# Model Results Section
with model_results:
    st.subheader('Prediction:')
    if user_text is not None and user_text.strip() != '':
        # Processing user_text
        # Removing punctuation
        user_text_clean = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        # Tokenizing
        custom_stop_words = set([  # Custom stop words defined here
            's', 'y', 'itself', 'this', 'to', 'after', 'yourselves', 'do', 'further', 'these', 'own', 
            # Truncated for brevity
        ])
        stop_words = set(stopwords.words('english')).union(custom_stop_words)
        tokens = nltk.word_tokenize(user_text_clean)
        # Removing stop words
        stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
        # Taking root word (lemmatization)
        lemmatizer = WordNetLemmatizer()
        lemmatized_output = [lemmatizer.lemmatize(word) for word in stopwords_removed]

        # Instantiating count vectorizer and loading pre-trained data
        stop_words = list(stop_words)  # Convert stop words to list for vectorizer
        count = CountVectorizer(stop_words=stop_words)
        X_train = pickle.load(open('pickle/X_train_2.pkl', 'rb'))
        X_test = lemmatized_output
        X_train_count = count.fit_transform(X_train)
        X_test_count = count.transform([' '.join(X_test)])  # Fix: Join tokens into a single string

        # Loading final model
        final_model = pickle.load(open('pickle/final_log_reg_count_model.pkl', 'rb'))

        # Apply model to make predictions
        prediction = final_model.predict(X_test_count)

        if prediction == 0:
            st.subheader('**Not Hate Speech**')
        else:
            st.subheader('**Hate Speech Detected 🚨**')  # Add a warning icon to make it visually attractive
    else:
        st.write("Please enter a valid tweet to get a prediction.")

# Sentiment Analysis Section
with sentiment_analysis:
    if user_text is not None and user_text.strip() != '':
        st.header('Sentiment Analysis with VADER')
        
        # Explaining VADER
        st.write("""*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
        st.text('')  # Spacer

        # Instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer() 
        # The object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text)
        # Determine sentiment category
        if sentiment_dict['compound'] >= 0.05: 
            category = "**Positive ✅**"
        elif sentiment_dict['compound'] <= -0.05: 
            category = "**Negative 🚫**" 
        else: 
            category = "**Neutral ☑️**"

        # Score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            st.write("Your Tweet is rated as", category) 
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            st.write("**Polarity Breakdown:**")
            st.write(sentiment_dict['neg'] * 100, "% Negative")
            st.write(sentiment_dict['neu'] * 100, "% Neutral")
            st.write(sentiment_dict['pos'] * 100, "% Positive")
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph)

# Contact Section
with contact:
    st.markdown("---")
    st.header("Contact")
    st.write("Connect with me on [LinkedIn](www.linkedin.com/in/priyangshu-chandra-das) for collaboration or inquiries.")
    st.write("You can also reach me via email at [pcdpcdjbx@gmail.com](mailto:pcdpcdjbx@gmail.com).")

# **[Twitter Hate Speech & Abusive Language Detection and Moderation Classifier](https://twitter-hate-speech-abusive-language-detection-and-moderation.streamlit.app/)**

![Cover](https://github.com/5PCD3/Twitter-Hate-Speech-Abusive-Language-Detection-and-Moderation-Classifier/blob/main/cover_picture.jpeg)
*This image is generated with NinjaTech AI*

## **Goal Of This Project**
Our goal is to create a binary classification model that detects slang words or hate speech so that these tweets can be automatically deleted.

## **Dataset Description**

The **Hate Speech and Offensive Language Dataset** is hosted on GitHub and is used for analyzing and classifying hate speech and offensive language. It is provided by [t-davidson](https://github.com/t-davidson) and includes:

- **Content**: Tweets categorized into hate speech, offensive language, and neither.
- **Purpose**: To train and evaluate models for detecting hate speech and offensive content on social media.
- **Data Source**: Twitter tweets, annotated for various types of abusive language.
- **Format**: CSV files with labeled examples for supervised learning tasks.

For more details and access to the dataset, visit [the GitHub repository](https://github.com/t-davidson/hate-speech-and-offensive-language).

## **Approach**

The `twiter_data.csv` file contains 5 columns:

* **`count`**: Number of CrowdFlower users who coded each tweet (min is 3; more users coded a tweet when judgments were determined to be unreliable).
* **`hate_speech`**: Number of CF users who judged the tweet to be hate speech.
* **`offensive_language`**: Number of CF users who judged the tweet to be offensive.
* **`neither`**: Number of CF users who judged the tweet to be neither offensive nor non-offensive.
* **`class`**: Class label for the majority of CF users. 0 - hate speech, 1 - offensive language, 2 - neither.

We created a binary classification model that detects slang words or hate speech by labeling:

- **Offensive language and hate speech** as `1` (to be deleted or blocked).
- **Comments that are neither offensive nor hate speech** as `0`.

This binary classification helps us efficiently determine which comments need moderation, allowing for the automatic deletion or blocking of inappropriate content.

**Steps Taken:**
1. **Exploratory Data Analysis (EDA)** and **Text Preprocessing**: Removed stopwords, lowercased text, removed punctuations, performed stemming, lemmatization, and tokenization.
2. **Feature Engineering**: Used Bag of Words (BOW), Bi-Grams, and TF-IDF techniques.
3. **Model Training**: Trained SVM, Logistic Regression, and Naive Bayes models. Found that SVM and Logistic Regression with TF-IDF, Bag of Words (BOW), and Bi-Grams provided similar F1 scores, while Naive Bayes performed comparatively worse.
4. **Model Evaluation**: Compared models using confusion matrices.

### Confusion Matrices

![SVM Confusion Matrix](https://github.com/5PCD3/Twitter-Hate-Speech-Abusive-Language-Detection-and-Moderation-Classifier/blob/main/plots/SVM_confusion_matrix_normalized.png)

![Logistic Regression Confusion Matrix](https://github.com/5PCD3/Twitter-Hate-Speech-Abusive-Language-Detection-and-Moderation-Classifier/blob/main/plots/Log_reg_confusion_matrix_normalized.png)

### Model Performance

- **Accuracy**: Both models have high accuracy, but Logistic Regression shows slightly better consistency across both classes with equal performance for each class (94%).
- **Class Balance**: Logistic Regression displays balanced performance for both classes, while SVM shows a trade-off between classes—better performance for class 1 but slightly worse for class 0.
- **Overall Performance**: Logistic Regression maintains high performance uniformly across both classes, resulting in a more consistent and balanced classifier.

### Why We Chose VADER for Sentiment Analysis

VADER (Valence Aware Dictionary and Sentiment Reasoner) was selected for sentiment analysis due to its effectiveness in analyzing social media text, which often contains informal language, slang, and emojis. Its strengths include:

- **Adaptability to Social Media**: VADER is specifically tuned to handle the nuances of social media text, making it well-suited for analyzing tweets.
- **Ease of Use**: It requires minimal preprocessing and is straightforward to implement, which streamlined the analysis process.
- **Accuracy**: VADER performs well in distinguishing sentiment in short texts like tweets, providing reliable sentiment scores for our model.

In conclusion, the Logistic Regression model with Bi-Grams features represents a better overall performance due to its balanced performance across both classes and higher true positive rates. Thus, we selected Logistic Regression and deployed it in a [Streamlit web application](https://twitter-hate-speech-abusive-language-detection-and-moderation.streamlit.app/). VADER was utilized for sentiment analysis to effectively handle the specific challenges of social media text.

**[Demo Video](https://github.com/5PCD3/Twitter-Hate-Speech-Abusive-Language-Detection-and-Moderation-Classifier/blob/main/TwitterHatespeechDetectionAppDemo.gif)** 
![Watch the Streamlit Web Application Demo](https://github.com/5PCD3/Twitter-Hate-Speech-Abusive-Language-Detection-and-Moderation-Classifier/blob/main/TwitterHatespeechDetectionAppDemo.gif)

**Note:**
*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*
# SocialMedia-Sentiment-Analysis

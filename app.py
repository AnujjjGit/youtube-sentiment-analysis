#import necessary libraries
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
import re
import nltk
import seaborn as sns
import missingno as msno
from obsei.source.youtube_scrapper import YoutubeScrapperSource, YoutubeScrapperConfig
from pandas import DataFrame
from obsei.sink.pandas_sink import PandasSink, PandasSinkConfig
from obsei.analyzer.classification_analyzer import (
    ClassificationAnalyzerConfig,
    ZeroShotClassificationAnalyzer,
)
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from wordcloud import WordCloud
from nltk.probability import FreqDist

nltk.download('stopwords')
nltk.download('punkt')
nltk.download("wordnet")
nltk.download('omw-1.4')
lemma = WordNetLemmatizer()
import streamlit as st
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
st.set_page_config(page_title="YouTube Comment Sentiment Analysis", page_icon="yt.png")
def set_custom_styles():
    with open("style copy.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

set_custom_styles()
image1 = "yt.png"  # Replace with the path to your image
# st.image(image1, use_column_width=True)
# st.title("YouTube Comment Sentiment Analysis") 
title_html = """
<div style="display: flex; align-items: center;">
    <img src="https://rb.gy/k5m0f" style="width: 95px; height: 70px; margin-right: 10px;" alt="Image">
    <h1>Youtube Comment Sentiment Analysis</h1>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)
video_url = st.text_input("Enter Your YouTube Video URL:")
if st.button("Analyze"):
    # Initialize the source configuration for pytchat
    source_config = YoutubeScrapperConfig(
        video_url= video_url,
        fetch_replies=False,
        max_comments = 84000,
        lookup_period="5Y",
    )
    source = YoutubeScrapperSource()
    source_response_list = source.lookup(source_config)
        # Create a ZeroShotClassificationAnalyzer instance
    text_analyzer = ZeroShotClassificationAnalyzer(
        model_name_or_path="typeform/mobilebert-uncased-mnli", device="auto"
    )
    # Analyze the input data using the ZeroShotClassificationAnalyzer
    # - It uses a pre-trained model specified by "model_name_or_path" to classify text.
    # - "device" is set to "auto" to automatically choose CPU or GPU.
    analyzer_response_list = text_analyzer.analyze_input(
        source_response_list=source_response_list,
        analyzer_config=ClassificationAnalyzerConfig(
            labels=["positive", "negative"],
        ),
    )
    # Configure the PandasSink to create a DataFrame
    sink_config = PandasSinkConfig(
        dataframe=DataFrame()
    )
    # Initialize a PandasSink instance
    sink = PandasSink()

    # Send the analyzer response data to the PandasSink for further processing and storage
    dataframe = sink.send_data(analyzer_response_list, sink_config)

    data = dataframe.iloc[:, :1]

    # Convert 'processed_text' column to lowercase
    data.loc[:, 'processed_text'] = data['processed_text'].str.lower()
    # Calculate the length of each text and create a new 'length' column
    data["length"] = data['processed_text'].apply(lambda x:len(x))

    # Tokenize the 'processed_text' and create a new 'tokenized_text' column
    data['tokenized_text']=data['processed_text'].apply(word_tokenize)

    # Define a function to remove stopwords from tokenized text
    def remove_stopwords(tokens):
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return filtered_tokens

    # Apply the 'remove_stopwords' function to the 'tokenized_text' column
    data['tokenized_text']= data['tokenized_text'].apply(remove_stopwords)


    # Define a function to clean text by removing special characters and URLs
    def clean_text(tokens):
        cleaned_tokens = []

        for token in tokens:
            # Remove special characters, single characters, and URLs
            cleaned_token = re.sub(r'<.*?>|[@#/]|[^\w\s]|^.$|http\S+', '', token)
            cleaned_tokens.append(cleaned_token)

        return cleaned_tokens


    # Define a function to cleanse the data by processing and cleaning the text
    def data_cleanse(comment):
        comment = comment.lower()
        comment = re.sub(r"@\S+", " ", comment)
        comment = re.sub(r"&.*?;|<.*?>", " ", comment)
        comment = re.sub(r"https?://\S+|www\.\S+", " ", comment)
        comment = re.sub(r"[^a-z]", " ", comment)
        comment = " ".join(word for word in word_tokenize(comment) if word not in stopwords.words('english'))
        comment = " ".join(lemma.lemmatize(word) for word in word_tokenize(comment))
        comment = re.sub(r"\b\w\b", "", comment).strip()

        return comment

    # Apply the 'data_cleanse' function to create a 'cleaned_text' column
    data['cleaned_text']=data['processed_text'].apply(data_cleanse)
    # Apply the 'clean_text' function to create a 'cleaned_column' column
    data['cleaned_column'] = data['tokenized_text'].apply(clean_text)

    # Combine all 'cleaned_text' values into a single string
    text1 = " ".join(title for title in data['cleaned_text'])

    # Create a WordCloud for visualizing word frequencies
    word_cloud1 = WordCloud(collocations = False, background_color = 'white',
                            width = 2048, height = 1080).generate(text1)

    # Define a function to get the polarity score of tokens
    def get_polarity_score(tokens):
        text = " ".join(tokens)  # Convert cleaned tokens back to a single string
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        return polarity

    # Calculate the polarity score for each 'cleaned_column' and create a 'polarity_score' column
    data['polarity_score'] = data['cleaned_column'].apply(get_polarity_score)
    
    # Define a function to label polarity based on polarity score
    def label_polarity(value):
        if value > 0:
            return 'pos'
        else :
            return 'neg'

    data['Values_Polarity']=data['polarity_score'].apply(label_polarity)

    label_counts1 = data['Values_Polarity'].value_counts()
    # label_counts
    custom_index_order = ['neg', 'pos']
    label_counts1 = label_counts1.reindex(custom_index_order, fill_value=0)
    print(label_counts1)

    st.subheader("Pie-chart :")
    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(label_counts1, labels=label_counts1.index, autopct='%1.1f%%', startangle=140, colors=['red', 'green'])
    # ax.pie(label_counts, labels=('pos','neg'), autopct='%1.1f%%', startangle=140, colors=['green', 'red'])

    ax.axis('equal')  # Equal aspect ratio ensures the pie is circular.


    # Add a title
    # ax.title('Distribution of Polarity Labels')

    # Display the pie chart
    # plt.show()
    st.pyplot(fig)

    # # Create a bar chart to visualize the distribution of polarity labels
    fig1,ax1=plt.subplots()
    st.subheader("Bar Graph :")
    # fig, ax = plt.subplots(figsize=(8, 6))
    label_counts1.plot(kind='bar', x='Polarity Labels', y='Count', color=['red', 'green'], ax=ax1)
    ax1.set_title('Distribution of Polarity Labels ')
    ax1.set_xlabel('Polarity Labels')
    ax1.set_ylabel('Count')
    # ax1.xticks(rotation=0)
    # ax1.set_xticklabels(label_counts['Polarity Labels'], rotation=0)  # Rotate x-axis labels if needed

    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig1)

    # Combine all 'cleaned_text' values into a single string for word frequency analysis
    all_comments = ' '.join(data['cleaned_text'].values)

    # Generate a WordCloud for the entire dataset
    st.subheader("Word Cloud:")
    wordcloud = WordCloud().generate(all_comments)
    wordcloud.to_file('woedcloud.png')
    st.image('woedcloud.png')
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")

    # Generate a WordCloud with a maximum font size of 40
    wordcloud1 = WordCloud(max_font_size=40).generate(all_comments)
    wordcloud1.to_file("wordcloud1.png")
    st.image('wordcloud1.png')

    # Plot the 12 most common words (cumulative=False)
    fig2,ax2=plt.subplots()
    st.subheader("Common Words v/s Frequency :")
    fdist = FreqDist(all_comments.split(' '))
    fdist.plot(12, cumulative=False)
    ax2.set_title('12 Most Common Words')
    ax2.set_xlabel('Words')
    ax2.set_ylabel('Frequency')

    st.pyplot(fig2)

st.markdown(
    """
    <div class="youtube-footer" style="text-align: center";>
        <p>&copy; 2023 YouTube, LLC</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.subheader("Disclaimer")
st.sidebar.write("This is a simplified YouTube comment sentiment analysis interface.")
st.sidebar.write("It uses the 'obsei' library to fetch YouTube comments and perform sentiment analysis.")
st.sidebar.write("For a full YouTube experience, visit youtube.com.")

# Instructions
st.sidebar.subheader("Instructions")
st.sidebar.write("1. Enter a valid YouTube video URL in the text input.")
st.sidebar.write("2. Click the 'Analyze Comments' button to view comments and sentiment analysis.")

# About
st.sidebar.subheader("About")
st.sidebar.write("This app is for demonstration purposes only.")
st.sidebar.write("Developed with Streamlit and 'obsei' library.")



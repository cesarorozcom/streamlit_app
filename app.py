import streamlit as st

st.set_page_config(page_title="NLPiffy", page_icon="🧊", layout="centered", initial_sidebar_state="auto")

from textblob import TextBlob
import spacy
import neattext as nt
from deep_translator import GoogleTranslator

from collections import Counter
import re

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from wordcloud import WordCloud


def summarize_text(text, num_sentences=3):
    # Remove special characters and convert text to lowercase
    clean_text = re.sub(r'\W', ' ', text.lower())

    # Split the text into words
    words = clean_text.split()

    # Calculate the frequency of each word
    word_freq = Counter(words)

    # Sort the words based on their frequency in descending order
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Extract the top 'num_sentences' most frequent words
    top_words = sorted_words[:num_sentences]

    # Create the summary by joining the top words
    summary = ' '.join(word for word, _ in top_words)

    return summary

@st.cache_data
# Lemma and Tokens Function
def text_analyzer(text):
    # import English library
    nlp = spacy.load("en_core_web_sm")
    # Create an nlp object
    docx = nlp(text)
    # Extract tokens and lemmas
    allData = [('"Tokens":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
    return allData

def main():
    """NLP App with Streamlit"""

    title_template = """
    <div stype="background-color:blue; padding:8px;">
    <h1 style="color:cyan">NLPiffy</h1>
    </div>
    """
    st.markdown(title_template, unsafe_allow_html=True)

    subheader_template = """
    <div style="background-color:cyan; padding:8px;">
    <h3 style="color:blue">Powered by Streamlit</h3>
    """

    st.markdown(subheader_template, unsafe_allow_html=True)

    st.title("Natural Language Processing App")

    activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
    st.sidebar.image("https://media.giphy.com/media/3o7TKz9bX9v9KzCvUQ/giphy.gif", use_column_width=True)
    choice = st.sidebar.selectbox("Menu", activity)

    if choice == "Text Analysis":
        st.subheader("Analysis of Text")
        st.write("This is a simple NLP app for text analysis")

        raw_text = st.text_area("Enter Text Here", "Type Here", height=300)

        if st.button("Analyze"):
            if len(raw_text) == 0:
                st.warning("Enter a text...")
            else:
                blob = TextBlob(raw_text)
                st.info("Basic functions")

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Basic Info"):
                        st.write("Text Stats")
                        word_desc = nt.TextFrame(raw_text).word_stats()
                        result_desc = {"Length of Text": word_desc["Length of Text"],
                                       "Num of Vowels": word_desc["Num of Vowels"],
                                       "Num of Consonants": word_desc["Num of Consonants"],
                                       "Num of Stopwords": word_desc["Num of Stopwords"], }
                        st.write(result_desc)
                    with st.expander("Stopwords"):
                        st.write("Stop words list")
                        stop_w = nt.TextExtractor(raw_text).extract_stopwords()
                        st.error(stop_w)

                with col2:
                    with st.expander("Processed Text"):
                        st.success("Stopwords Excluded Text")
                        processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
                        st.write(processed_text)

                    with st.expander("Plot Wordcloud"):
                        st.success("Wordcloud")
                        wordcloud = WordCloud().generate(raw_text)
                        fig = plt.figure(1, figsize=(20, 10))
                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        st.pyplot(fig)
                st.write("")
                st.write("")
                st.info("Advanced Features")

                col3, col4 = st.columns(2)

                with col3:
                    with st.expander("Tokens&Lemmas"):
                        st.write("T&K")
                        processed_text_mid = str(nt.TextFrame(raw_text).remove_stopwords())
                        processed_text_mid = str(nt.TextFrame(processed_text_mid).remove_puncts())
                        processed_text_fin = str(nt.TextFrame(processed_text_mid).remove_special_characters())
                        tandl = text_analyzer(processed_text_fin)
                        st.json(tandl)
                with col4:
                    with st.expander("Summarize"):
                        st.success("Summarize")
                        summary = summarize_text(raw_text)
                        st.success(summary)

    if choice == "Translation":
        st.subheader("Translation")
        st.write(" ")
        st.write(" ")
        raw_text = st.text_area("Original text", "Write something to be translated...", height=200)
        if len(raw_text) < 3:
            st.warning("Please enter a text with at least 3 characters")
        else:
            target_language = st.selectbox("Select Language", ["German", "Spanish", "French", "Chinese", "Japanese"])
            if target_language == "German":
                language_code = "de"
            elif target_language == "Spanish":
                language_code = "es"
            elif target_language == "French":
                language_code = "fr"
            elif target_language == "Chinese":
                language_code = "zh-cn"
            elif target_language == "Japanese":
                language_code = "ja"

            if st.button("Translate"):
                with st.spinner('Wait for it...'):
                    translator = GoogleTranslator(source="auto", target=language_code).translate(raw_text)
                    translated_text = translator.translate(raw_text)
                    st.write(translated_text)

    if choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        st.write(" ")
        st.write(" ")
        raw_text = st.text_area("Text to analyse", "Enter a text here...", height=200)
        if st.button("Evaluate"):
            if len(raw_text) == 0:
                st.warning("Please enter a text...")
            else:
                blob = TextBlob(raw_text)
                st.info("Sentiment Analysis")
                st.write(blob.sentiment)
                st.write(" ")
    if choice == "About":
        st.subheader("About")
        st.write("")

        st.markdown("""
        ### Built with Streamlit
        
        for info:
        - [Streamlit](https://streamlit.io/)
        """)


if __name__ == "__main__":
    main()

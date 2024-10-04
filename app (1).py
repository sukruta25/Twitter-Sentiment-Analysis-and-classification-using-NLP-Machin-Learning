
import streamlit as st
import pandas as pd
import re
import unicodedata
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import pickle

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4') 
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)  # Required for word_tokenize

# Define the CleaningPipeline class
class CleaningPipeline:
    def __init__(self):
        pass

    def remove_non_ascii(self, text):
        """Remove non-ASCII characters from the text"""
        text = re.sub(r'\x85', '', text)  
        text = re.sub(r'\x91', '', text)  
        text = re.sub(r'\x92', '', text)  
        text = re.sub(u'\x93', '', text)  
        text = re.sub(u'\x94', '', text)  
        text = re.sub(r'\x95', '', text)  
        text = re.sub(r'\x96', '', text)  
        text = re.sub(r'\x99', '', text)  
        text = re.sub(r'\xae', '', text)  
        text = re.sub(r'\xb0', '', text)  
        text = re.sub(r'\xba', '', text)  
        new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return new_text

    def remove_whitespace_and_special_chars(self, text):
        '''Remove whitespace and special characters from the text'''
        tab_newline_pattern = '[\t\n]'  
        multi_space = ' {2,}'  

        formatted_text = text.lower()  
        formatted_text = re.sub(r'(?:\d{1,2} )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (?:\d{1,2}, )?\d{2,4}', 'mdate', formatted_text)
        formatted_text = re.sub(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}[a-z]*', 'mdate', formatted_text)
        formatted_text = re.sub('(?<=\d),(?=\d)', 'commaseperatednum', formatted_text)
        formatted_text = re.sub(tab_newline_pattern, ' ', formatted_text)
        formatted_text = re.sub(multi_space, ' ', formatted_text)
        formatted_text = re.compile(r'[^a-zA-Z0-9\s]').sub(' ', formatted_text)
        formatted_text = re.sub(multi_space, ' ', formatted_text)
        return formatted_text

    def remove_numerics(self, text):
        '''Remove all numeric values from the text'''
        text = re.sub(r'\d+', '', text)
        return text

    def removeStopWord(self, text):
        '''Removes all stopwords e.g., a, the, etc...'''
        stop = set(stopwords.words('english'))
        return " ".join([word for word in word_tokenize(text) if word.lower() not in stop])

    def word_lemmatization(self, text):
        '''Lemmatize words in the text'''
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        lemmatized_text = " ".join(lemmatized_words)
        return lemmatized_text

    def transform(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = self.remove_non_ascii(text)
        text = self.remove_whitespace_and_special_chars(text)
        text = self.removeStopWord(text)
        text = self.word_lemmatization(text)
        text = self.remove_numerics(text)
        return text

def identify_text_column(df):
    """Automatically identify the text column based on content."""
    max_words = 0
    text_column = None

    for column in df.columns:
        # Check if the column is a string type
        if df[column].dtype == object:
            # Calculate the average number of words in the column
            avg_words = df[column].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0).mean()
            if avg_words > max_words:
                max_words = avg_words
                text_column = column

    return text_column

def main():
    st.title("Text Processing and Classification")

    # Load the combined vectorization and model pipeline
    with open('vectorization_model_pipeline_1.pkl', 'rb') as file:
        pipeline = pickle.load(file)

    # Load the BERT model for topic classification
    with open('bert_model.pkl', 'rb') as file:
        bert_model = pickle.load(file)

    # Add a sidebar for selecting the functionality
    option = st.sidebar.selectbox("Choose the functionality", ("Upload CSV", "Enter Text"))

    if option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Automatically identify the text column
            text_column = identify_text_column(df)

            if text_column is None:
                st.warning("No suitable text column found in the uploaded file.")
                return

            st.write(f"Detected text column: {text_column}")
            st.write(df[text_column])

            st.header("Text Data")
            st.subheader("Processing Text Data")

            # Handle NaN values
            df[text_column] = df[text_column].fillna('')

            # Clean the text data
            cleaning_pipeline = CleaningPipeline()
            df['cleaned_text'] = df[text_column].apply(cleaning_pipeline.transform)

            # Use the loaded pipeline to make predictions
            df['predictions'] = pipeline.predict(df['cleaned_text'])

            # Replace 0 and 4 with -ve and +ve sentiment
            df['predictions'] = df['predictions'].replace({0: 'Negative', 4: 'Positive'})

            # Use the BERT model to predict topics
            df['topic'] = bert_model.predict(df['cleaned_text'])

            # Display the results
            st.subheader("Processed Data with Predictions and Topics")
            st.write(df[[text_column, 'cleaned_text', 'predictions', 'topic']])

            # Option to download the processed DataFrame as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed Data", csv, "processed_data.csv", "text/csv")

    elif option == "Enter Text":
        user_input = st.text_area("Enter your text here:")
        if st.button("Analyze Sentiment and Topic"):
            if user_input:
                # Clean the input text
                cleaning_pipeline = CleaningPipeline()
                cleaned_text = cleaning_pipeline.transform(user_input)

                # Use the loaded pipeline to make predictions
                prediction = pipeline.predict([cleaned_text])

                # Replace 0 and 4 with -ve and +ve sentiment
                sentiment = 'Negative' if prediction[0] == 0 else 'Positive'

                # Use the BERT model to predict the topic
                topic = bert_model.predict([cleaned_text])[0]

                st.subheader("Analysis Result")
                st.write(f"Sentiment: {sentiment}")
                st.write(f"Topic of discussion: {topic}")
            else:
                st.warning("Please enter some text for analysis.")

if __name__ == '__main__':
    main()

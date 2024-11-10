import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv("product_reviews.csv")

# Sentiment analysis function (for manual labeling or fallback)
def analyze_sentiment(review):
    try:
        if not isinstance(review, str):
            review = str(review)  # Convert non-strings to strings
        
        sentiment = TextBlob(review).sentiment.polarity
        if sentiment > 0.1:
            return 'positive'
        elif sentiment > 0.05:
            return 'neutral'
        else:
            return 'negative'
    except Exception as e:
        print(f"Error processing review: {review}. Error: {e}")
        return 'neutral'

# Apply sentiment analysis to each review if no labels are available
def preprocess_data(data):
    # Handle NaN values in the reviews column by filling them with an empty string
    data['reviews'] = data['reviews'].fillna('')
    
    # Alternatively, you could drop rows with NaN in the 'reviews' column:
    # data = data.dropna(subset=['reviews'])
    
    data['true_sentiment'] = data['reviews'].apply(analyze_sentiment)
    data['reviews_clean'] = data['reviews'].str.lower()  # Clean the review text
    return data

# Function to prepare training and testing data
def prepare_data(data):
    X = data['reviews_clean']
    y = data['true_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Vectorizing the reviews using TF-IDF
def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

# Train the model (Logistic Regression in this case)
def train_model(X_train_vec, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
    
    return accuracy, precision, recall, f1, conf_matrix

# Function to display results in Streamlit
def display_results(accuracy, precision, recall, f1, conf_matrix, top_products_per_category):
    st.title("Product Review Analysis Using Custom Model")
    
    st.subheader("Model Performance Metrics:")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    
    st.subheader("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)
    
    st.subheader("Top 5 Products in Each Category by Sentiment Score")
    st.write(top_products_per_category)

# Main function to integrate everything
def main():
    # Load and preprocess data
    data = load_data()  # This ensures data is correctly loaded
    data = preprocess_data(data)  # Preprocess the data
    
    # Prepare the data for training/testing
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Vectorize the text data
    X_train_vec, X_test_vec, vectorizer = vectorize_data(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_vec, y_train)
    
    # Evaluate the model
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test_vec, y_test)
    
    # Calculate top 5 products by sentiment score for each category
    data['sentiment_score'] = data['reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)
    product_avg_sentiment = data.groupby(['categories', 'name']).agg({'sentiment_score': 'mean'}).reset_index()
    top_products_per_category = product_avg_sentiment.groupby('categories').apply(lambda x: x.nlargest(5, 'sentiment_score')).reset_index(drop=True)
    
    # Display results on Streamlit
    display_results(accuracy, precision, recall, f1, conf_matrix, top_products_per_category)

# Run the app
if __name__ == "__main__":
    main()

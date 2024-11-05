# Sentiment Prediction of Amazon-Alexa-Reviews using Flask 

**Special Thanks to @Satyajit Pattnaik for his deliberate work here and educational videos on Flask.**

This repository contains a sentiment prediction web application that provides a frontend with Flask API. The app accepts both forms of input as a single text or a CSV file with multiple sentences, and outputs predictions indicating positive or negative sentiment.

## Table of Contents

1. Features
2. Project Structure
3. Setup and Installation
4. Using the Flask API
5. API Endpoints
6. NLP Steps and EDA
7. Acknowledgments
8. Future Work


## Features

- **Single text prediction**: Predict the sentiment of individual text inputs through the Flask API interface
- **Bulk text prediction**: Upload a CSV file with a "Sentences" column for batch sentiment predictions.
- **Visual Graphing**: Visualizes the distribution of sentiments from bulk predictions
- **Downloadable CSV**: Allows users to download the prediction results as a "predictions" CSV file.

## Project Structure


├── api.py               # Main Flask app for serving predictions via API<br/> 
├── EDA & Modeling.ipynb # Our Data Science workflow is inside<br/>
├── Data/<br/>
 |   ├── amazon_alexa.tsv #This is the data we train our model on<br/>
├── Models/              # Folder containing model, scaler, and vectorizer files<br/>
│   ├── model_xgb.pkl<br/>
│   ├── scaler.pkl<br/>
│   └── countVectorizer.pkl<br/>
 ├── templates/<br/>
│   └── landing.html     # HTML template for the Flask app's main page<br/>
└── README.md            # Project documentation (you are here)<br/>

## Setup and Installation

**Prerequisites**
- Python 3.8+
- Flask 

## Step-by-Step Setup

1. **Clone the repository**

git clone https://github.com/groovyds/sentiment-prediction-app.git

cd sentiment-prediction-app

2. **Setup Virtual Environment**

- Create and activate the new virtual environment

python -m venv env

env\Scripts\activate

- Install dependencies from requirements.txt

pip install -r requiremnets.txt

(Optional) Set up a separate environment for Machine Learning

3. **Run Flask**

In the terminal of api.py file, "python api.py"


The app will run on port 5000. 
Debug=False after you're done developing the app!

## **Usage**

**Using Flask API**

1. Start the Flask server by running
    python api.py in the IDE terminal

2. Write/Upload requests to the /predict endpoint
    - **Single Prediction:** Send a POST request with a JSON object containing a text field
    - **Bulk Prediction:** Send a POST request with a CSV file under the file parameter in form-data

## **API Endpoints**
**POST** /predict
- **Description:** Returns a prediction for the provided input.
- **Parameters:** 
    - **File:**CSV file containing sentences for bulk prediction with a "Sentences" column
    - **Text:** Single text input sentence for prediction
- **Responses:** Returns predictions in JSON format or as a downloadable CSV file

# NLP Steps & EDA

## **NLP Steps**

The initial objective of this project was to use Flask in a more intermediate way and not focus on the NLP methods and techniques. We will reference areas where our steps can be changed or objectively improved in the Future Work section.

**Data Preparation**

- Import our data from a TSV format delimiter is "t"
- Drop any null values 
- We created a new "lenght" column to analyze and discover new patterns and meaning behind our review data where our lenght column is the number of characters in the review

**EDA**

- This section contains numerous graphs to better visualize and understand our dataset.
    - One of the more interesting graphs is the difference of length between positive and negative reviews
- Also in this section we uncover that our dataset is unbalanced woth 92% positive reviews and 8% negative reviews
- Another interesting graph is drawn from the variation of sales of different colours of our product and the "Black Dot" is the most prominent in this area.

### **Cloud of Words**

- We visualised the most common words (50 words to be exact) in both negative and positive reviews to understand the common usage of words

## **Preprocessing**

We used a simpler method of preparing our corpus since this project's main objective revolves around the usage of Flask, so we decided to use **Porter Stemmer** because of it's simpler nature and faster processing times. This can be a good starting point to see how much we can increase our performance.

After creating our corpus we decided to use **Count Vectorizer** to fit and transform our dataset, this method fits to our objective since this will be a starting point.

## **Train-Test Split**

Divides our dataset of X and y into X_train, X_test, y_train and y_test

## **Model Selection**

- We evaluated and tested the performances of three different models; Decision Trees, Random Forest and XGBoost Classifier with Accuracy Scores and Confusion Matrices. 

- **Problem:** At an initial glance they seem to be working fine until we realize the performances are around 92% which is problematic since our dataset is imbalanced and we didn't address the issue in this project. We chose XGBoost Classifier since it had the best performance overall.

- The best performing model for this project was the XGBoost Classifier with 94% Accuracy on the test set

- We exported our trained model to a Pickle file for later use on our Flask app.

# **Acknowledgments**

This project was developed using **Flask** on backend and frontend operations for a user-friendly interface. The sentiment analysis model is based on the machine learning models and is trained and exported out in a Pickle format including an XGBoost Classifier and custom preprocessing steps for NLP.

# **Future Work**

For better performance we can employ different steps in the future like TF-IDF or completely different RNN or LSTM models, or even more sophisticated transfer-learning models like BERT

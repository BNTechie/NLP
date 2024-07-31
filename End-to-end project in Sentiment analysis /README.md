Let's go through an end-to-end project on Natural Language Processing (NLP). We'll build a sentiment analysis application using a simple machine learning model. We'll use the IMDb movie reviews dataset to train our model, create a FastAPI application to serve predictions, and deploy it using Docker.



# Sentiment Analysis with IMDb Movie Reviews

This project demonstrates an end-to-end sentiment analysis application using a machine learning model. The steps include data collection and preprocessing, model training and saving, building a FastAPI application, creating and running a Docker container, and cleaning up Docker resources.

## Table of Contents

1. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
2. [Model Training and Saving](#model-training-and-saving)
3. [Building a FastAPI Application](#building-a-fastapi-application)
4. [Creating and Running a Docker Container](#creating-and-running-a-docker-container)
5. [Cleaning Up Docker Resources](#cleaning-up-docker-resources)


## Step 1: Data Collection and Preprocessing

Install Required Libraries:


```
pip install pandas scikit-learn fastapi uvicorn pydantic

```


## Download and Preprocess Data:

Create a script data_preprocessing.py:

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
df = pd.read_csv(url, compression='gzip', header=0, sep='\t', quotechar='"')

# Sample for quick processing
df = df.sample(10000, random_state=42)

# Split data
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

```



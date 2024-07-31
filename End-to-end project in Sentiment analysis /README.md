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

Run the script:

```
python data_preprocessing.py

```
### Step 2: Model Training and Saving
#### Train and Save the Model:

Create a script train_model.py:


```
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression())
])

# Train the model
pipeline.fit(X_train.squeeze(), y_train.squeeze())

# Save the model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully!")


```


Run the script:
```
python train_model.py

```

### Step 3: Building a FastAPI Application

Create app.py:

```
**from fastapi import FastAPI, Request
import pickle
from pydantic import BaseModel
from typing import List

class Review(BaseModel):
    review: str

# Load the model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(review: Review):
    prediction = model.predict([review.review])
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return {"sentiment": sentiment}

```

Install FastAPI and Uvicorn:

```
pip install fastapi uvicorn

```

Run the FastAPI App:
```
uvicorn app:app --reload

```
Test the API:

Use a tool like Postman or curl to send a POST request to your running API:

```
POST http://localhost:8000/predict
Content-Type: application/json

{
  "review": "This movie was absolutely wonderful, I loved it!"
}

```







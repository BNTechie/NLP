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

´´´       



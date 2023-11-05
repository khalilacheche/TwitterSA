# Story
This project was done as part of the course "Machine Learning" at EPFL. The project was self-guided and done in collaboration with 2 other students (Mehdi Mezghani and Youssef Mamlouk).

# What
This project's primary objective was sentiment analysis of tweets. It involved classifying tweets as either positive or negative based on the presence of specific emojis. The team worked with a dataset containing 2.5 million tweets, with a balanced distribution of 50% positive and 50% negative sentiments.

# How
To achieve our goal, we employed a variety of machine learning techniques, including NLP methods. The most notable achievement was the development of a fine-tuned model based on BERT, a state-of-the-art transformer-based model for NLP tasks.

The details of the approach and methods are thoroughly discussed in the provided PDF report of the git repository.

# Challenges

Data Preprocessing: Preparing the large dataset of tweets for model training required significant data cleaning and preprocessing efforts, which are documented in the "data_cleaning.ipynb" notebook.

Model Development: Developing and fine-tuning deep learning models, including BERT-based models, neural networks with Word2Vec embeddings, and logistic regression models with TF_IDF embeddings, involved substantial experimentation and iterations.

Training the models: Training the fine-tuned model required significant computational resources and time. We used Google Colab to train the models.

Reproducibility: Ensuring that others could reproduce their results and retrain the models required detailed instructions and the provision of datasets and model files.

Result
We achieved an impressive 91.4% accuracy on [ai_crowd](https://www.aicrowd.com/challenges/epfl-ml-text-classification/leaderboards?challenge_round_id=1251).
# Emotion-Based Chatbot

This project is a machine learning-powered chatbot that analyses the emotion of a user’s input and responds with sympathetic messages or/and guidance based on the detected emotion and the style.

Features:

Emotion Classifier – Labels the input sentence with one of six emotions: sadness, joy, fear, anger, love, surprise.
Pre-trained Language Model – Generates a response tailored to the detected emotion.


# Installation

1. Clone the repository 
    `git clone <repo link>`
    `cd <repo folder>`
2. Set up and activate your virtual environment
    `python -m venv venv`
    `source venv/bin/activate`
3. Install dependencies in requirements.txt
    `pip install -r requirements.txt`

# Usage

Run `python main.py`.
Enter your input and get a response.


# Citation Information 
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}

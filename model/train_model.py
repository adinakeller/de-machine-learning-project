import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_csv(file_name):
    df = pd.read_csv(file_name)
    return df

def train_data(df):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    vr = TfidfVectorizer()
    X_train_vec = vr.fit_transform(X_train)
    X_test_vec = vr.transform(X_test)

    logreg = LogisticRegression()
    logreg.fit(X_train_vec, y_train)
    y_pred = logreg.predict(X_test_vec)

    return y_pred, y_test


def save_trained_model(model, file_name):
    with open(file_name,'wb') as f:
        trained_model = pickle.dump(model, f)

    logger.info(f'Model saved to {file_name}')
    return trained_model

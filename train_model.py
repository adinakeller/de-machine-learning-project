import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_csv(file_name):
    df = pd.read_csv(file_name)
    X = df['text']
    y = df['label']
    return X, y


def train_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

    vr = TfidfVectorizer()
    X_train = vr.fit_transform(X_train)
    X_val = vr.transform(X_val)

    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    logger.info(f'Confusion Matrix: {cm}')

    target_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    cr = classification_report(y_val, y_pred, target_names=target_names)
    logger.info(f'Classification Report: {cr}')

def save_trained_model(model, file_name):
    with open(file_name,'wb') as f:
        pickle.dump(model, f)

    logger.info(f'Model saved to {file_name}')
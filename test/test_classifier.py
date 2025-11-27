from model.classifier import EmotionClassifier
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def test_loads_trained_model(tmp_path):
    model = LogisticRegression()

    file = tmp_path / 'model.pkl'
    with open(file,'wb') as f:
        pickle.dump(model, f)

    c = EmotionClassifier()
    result = c.load_trained_model(file)

    assert isinstance(result, LogisticRegression)


def test_loads_vectorizer_model(tmp_path):
    v = TfidfVectorizer()

    file = tmp_path / 'model.pkl'
    with open(file,'wb') as f:
        pickle.dump(v, f)

    c = EmotionClassifier()
    result = c.load_vectorizer(file)

    assert isinstance(result, TfidfVectorizer)

def test_returns_correct_emotion():
    emotions = {
            0: 'sadness',
            1: 'joy',
            2: 'love'
        }
    c = EmotionClassifier()
    result = c.convert_to_emotion(0)
    
    assert result == 'sadness'

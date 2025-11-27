import pandas as pd

class EmotionClassifier:

    def load_vectorizer(self, file):
        self.vectorizer = pd.read_pickle(file)
        return self.vectorizer

    def load_trained_model(self, file):
        self.trained_model = pd.read_pickle(file)
        return self.trained_model

    def classify_emotion(self, sentence: str):
        v = self.vectorizer.transform(sentence)
        prediction = self.trained_model.predict(v)
        return prediction
    
    def convert_to_emotion(self, prediction: int):
        emotions = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'suprise'
        }
        return emotions[prediction]

    
# c = EmotionClassifier()
# c.load_vectorizer('pickle_files/vectorizer.pkl')
# c.load_trained_model('pickle_files/trained_model.pkl')
# pred = c.classify_emotion(['i am happy', 'i hate that'])
# print(pred)
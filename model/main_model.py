from model.train_model import load_csv, train_data
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import logging 
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

df = load_csv('cleaned.csv')

y_pred, y_test = train_data(df)

cm = confusion_matrix(y_test, y_pred)
logger.info(f'Confusion Matrix: {cm}')

target_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
cr = classification_report(y_test, y_pred, target_names=target_names)
logger.info(f'Classification Report: {cr}')

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.3f}')

# with open("accuracy.json", "w") as f:
#     json.dump({"accuracy": accuracy}, f)


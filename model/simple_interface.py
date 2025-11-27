from classifier import EmotionClassifier

c = EmotionClassifier()
c.load_vectorizer('pickle_files/vectorizer.pkl')
c.load_trained_model('pickle_files/trained_model.pkl')

print('Welcome! Enter a sentence to classify!\n')
print('To end the program enter "exit"\n')

while True:
    user_input = input('Enter a sentence: ')
    pred = c.classify_emotion([user_input])

    if user_input == 'exit':
        break
    else:
        reply = c.convert_to_emotion(pred[0])
        print(f'Emotion: {reply}\n')
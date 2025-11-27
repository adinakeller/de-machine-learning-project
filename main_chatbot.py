from chatbot.chatbot_interface import Chatbot
import json

bot = Chatbot()

print('Welcome! Enter a sentence to get an emotion label!\n')
print('To end the program enter "exit"\n')

while True:
    user_input = input('Assitant: What would you like me to classify?\nYou: ')
    reply = bot.generate_reply(user_input)
    emotion = bot.pass_input_into_classifier(reply)
    final = bot.final_reply(user_input, emotion)
  
    if user_input == 'exit':
        break
    else:
        print(f'Assistant: Emotion label {final}\n')
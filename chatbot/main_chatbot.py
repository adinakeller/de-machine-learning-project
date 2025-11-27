from chatbot.chatbot_interface import Chatbot

bot = Chatbot()

print('Welcome! Enter a sentence to get an emotion label!\n')
print('To end the program enter "exit"\n')

while True:
    user_input = input('Assitant: What would you like me to classify?\nYou: ')
    format = f"<|user|> {user_input} <|end|> <|assistant|>"
    reply = bot.generate_reply(format)
    emotion = bot.pass_input_into_classifier(reply)
    final = bot.final_reply(format, emotion)
  
    if user_input == 'exit':
        break
    else:
        print(f'Assistant: Emotion label {final}\n')
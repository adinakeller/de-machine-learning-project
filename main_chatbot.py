from chatbot.chatbot_interface import Chatbot

bot = Chatbot()

print('Welcome! Let\'s chat!\n')
print('To end the program enter "exit"\n')
print('Assistant: What are you feeling today?')

while True:
    user_input = input('You: ')
    reply = bot.generate_reply(user_input)
    emotion = bot.pass_input_into_classifier(reply)
    final = bot.final_reply(user_input, emotion)
  
    if user_input == 'exit':
        break
    else:
        print(f'Assistant: {final}\n')
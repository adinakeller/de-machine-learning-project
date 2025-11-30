from chatbot.chatbot_interface import Chatbot

bot = Chatbot()

print('Welcome! Tell me your thoughts and i will provide some emotional insight.\n')
print('To end the program enter "exit"\n')
#style = print(input('Choose how you want me to repsond (sarcastic, formal, overly enthusiastic or bored office worker): \n'))

while True:
    user_input = input('You: ')
    reply = bot.generate_reply(user_input)
    emotion = bot.pass_input_into_classifier(reply)
    final = bot.final_reply(user_input, emotion)
  
    if user_input == 'exit':
        break
    else:
        print(f'Assistant: {final}\n')
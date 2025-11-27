from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from model.classifier import EmotionClassifier

class Chatbot:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device} acceleration\n')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.system_prompt = """
            You are a data exctracting AI. You MUST extract ONLY the part of the users input that conveys emotion.

            You MUST output valid JSON in the following format:

            {"extracted_data": "the exact data you extracted from the context you used, copied word-for-word"}

            If there is NO data that conveys emotion, output:
            {"extracted_data": ""}

            ===========================
            EXAMPLES (FOLLOW THESE EXACTLY)
            ===========================

            QUESTION: What are they feeling when my friend says 'i am happy'?
            OUTPUT:
            {"extracted_data": "happy"}

            QUESTION: Now i will start to feel resentful.
            OUTPUT:
            {"extracted_data": "resentful"}

            QUESTION: when are we leaving?
            OUTPUT:
            {"extracted_data": ""}
            """

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:

        encode = self.encode_prompt(prompt + '\n' + self.system_prompt)

        reply = self.model.generate(
            input_ids=encode["input_ids"], 
            attention_mask=encode['attention_mask'], 
            pad_token_id=self.tokenizer.eos_token_id, 
            do_sample=True,
            max_new_tokens=1000, 
            top_p=0.95, 
            top_k=50,
            temperature=0.9
            )

        decode = self.decode_reply(reply[0])

        return decode
    
    def pass_input_into_classifier(self, input: str):
        c = EmotionClassifier()
        c.load_vectorizer('pickle_files/vectorizer.pkl')
        c.load_trained_model('pickle_files/trained_model.pkl')
        pred = c.classify_emotion([input])
        emotion = c.convert_to_emotion(pred[0])

        return emotion
    
    def final_reply(self, user_prompt: str, emotion: str):
        prompt = f'''
        You are a sympathetic assistant. 

        You should reply to the user in a warm, friendly and helpful way.

        What the user said: {user_prompt}
        The classified emotion from what the user said: {emotion}
        '''
        encode = self.encode_prompt(prompt)

        reply = self.model.generate(
            input_ids=encode["input_ids"], 
            attention_mask=encode['attention_mask'], 
            pad_token_id=self.tokenizer.eos_token_id, 
            do_sample=True,
            max_new_tokens=1000, 
            top_p=0.95, 
            top_k=50,
            temperature=0.9
            )

        decode = self.decode_reply(reply[0])

        return decode
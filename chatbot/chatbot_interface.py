from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from model.classifier import EmotionClassifier

class Chatbot:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device} acceleration\n')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.system_prompt = """
            <|system|>\nYou are a data exctracting AI. You MUST extract ONLY the part of the users input that conveys emotion.

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
            {"extracted_data": ""}<|end|>\n
            """

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:

        encode = self.encode_prompt(prompt + '\n')

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

    def final_reply(self, style: str, user_prompt: str, emotion: str):
        system_prompt = f"""
        You are a {style} assistant who explains the results of an emotion classification to the user in natural language.

        ===========================
        EXAMPLES (FOLLOW THESE EXACTLY)
        ===========================

        STYLE: pirate
        USER: i feel pretty pathetic most of the time
        YOUR RESPONSE: Arrr, sounds like ye be feelin' {emotion}, matey. That be a rough sea to sail. Here be a few bits of wisdom to help steady yer ship…
        
        STYLE: overly enthusiastic
        USER: im on a boat trip to denmark with my friends!
        YOUR RESPONSE: Sounds like you are feeling {emotion}!! THAT SOUNDS AMAZING!!! I hope you're having the BEST time ever!!!
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        encode = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.encode_prompt(encode)

        reply = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=self.tokenizer.eos_token_id, 
            do_sample=True,
            max_new_tokens=1500,
            top_p=0.95, 
            top_k=50,
            temperature=0.7
        )

        new_token = reply[0][inputs['input_ids'].size(1) :]
        decode = self.decode_reply(new_token)

        return decode


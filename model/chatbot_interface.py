from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device} acceleration\n')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.system_prompt = f"""
            You are a data exctracting AI. You MUST extract ONLY the part of the users input that conveys emotion.

            You MUST output valid JSON in the following format:

            {
            "extracted_data": "the exact sentence or word you extracted from the context you used, copied word-for-word"
            }

            If there is NO data that conveys emotion, output:
            {
            "extracted_data": ""
            }

            ===========================
            EXAMPLES (FOLLOW THESE EXACTLY)
            ===========================

            QUESTION: What are they feeling when my friend says 'i am happy'?
            OUTPUT:
            {
            "extracted_data": "i am happy",
            }

            QUESTION: when are we leaving?
            OUTPUT:
            {
            "extracted_data": "",
            }
            """

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
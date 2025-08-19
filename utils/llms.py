from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class RagLLM1:
    def __init__(self):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained("./LLM/meta-llama/meta-llama-3-8b-instruct",
                                                          torch_dtype=torch.bfloat16, device_map="auto").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("./LLM/meta-llama/meta-llama-3-8b-instruct")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        


    def invoke(self, inputs: str,max_tokens) -> str:
      
        prompt = inputs
        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = model_inputs.input_ids.to(self.device)
        attention_mask = model_inputs.attention_mask.to(self.device)
        
        generate_kwargs_not_stream = {
            "inputs": input_ids,
            "temperature": 1,
            "max_new_tokens": max_tokens,
            "attention_mask": attention_mask,
            "eos_token_id": self.model.config.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        output_tensor = self.model.generate(**generate_kwargs_not_stream)  
        outputs = output_tensor[:, input_ids.shape[1]:]
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response



class RagLLM2:
    def __init__(self):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained("./LLM/mistralai/mistral-7b-instruct-v0-2"
                                                          ,torch_dtype=torch.bfloat16).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("./LLM/mistralai/mistral-7b-instruct-v0-2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))


    def invoke(self, inputs: str,max_tokens) -> str:
        
        prompt = inputs
        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = model_inputs.input_ids.to(self.device)
        attention_mask = model_inputs.attention_mask.to(self.device)
        
        generate_kwargs_not_stream = {
            "inputs": input_ids,
            "temperature": 1,
            "max_new_tokens": max_tokens,
            "attention_mask": attention_mask,
            "eos_token_id": self.model.config.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        output_tensor = self.model.generate(**generate_kwargs_not_stream)  
        outputs = output_tensor[:, input_ids.shape[1]:]
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response



class RagLLM3:
    def __init__(self):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(
            "LLM/glm-4-9b-chat", torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "LLM/glm-4-9b-chat", trust_remote_code=True, encode_special_tokens=True, use_fast=False
        )


    def invoke(self, inputs: str, max_tokens) -> str:
        
        prompt = inputs
        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = model_inputs.input_ids.to(self.device)
        attention_mask = model_inputs.attention_mask.to(self.device)
        
        generate_kwargs_not_stream = {
            "inputs": input_ids,
            "temperature":1,
            "max_new_tokens": max_tokens,
        }

        output_tensor = self.model.generate(**generate_kwargs_not_stream)  
        outputs = output_tensor[:, input_ids.shape[1]:]
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


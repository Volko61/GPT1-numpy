from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import torch

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
# LMHeadModel includes the text generation layers
model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# Generate next tokens automatically
output_ids = model.generate(**inputs, max_length=50, do_sample=True)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
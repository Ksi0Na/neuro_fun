import gpt_2_simple as gpt2
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
import os

model_path = "gpt2/models/ru/774M"
model_name = 'sberbank-ai/rugpt2large'

your_text = 'Привет!'

if not os.path.exists(model_path):
    print(f"Downloading gpt-2 {model_name} model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

generator = pipeline(task='text-generation',
                     model=model_path,
                     tokenizer=model_path)

gen_text = generator(text_inputs=your_text,
                     # prompt=your_text,
                     max_length=150,
                     do_sample=True,
                     temperature=0.7)[0]['generated_text']

print(gen_text)

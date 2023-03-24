from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
import os

# model_path = "gpt3/ru/124M"
# model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

model_path = "gpt3/ru/355M"
model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"

# model_path = "gpt3/ru/774M"
# model_name = "sberbank-ai/rugpt3large_based_on_gpt2"

your_text = "Привет"

if not os.path.exists(model_path):
    print(f"Downloading gpt-2 {model_name} model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

generator = pipeline(task='text-generation',
                     model=model_path,
                     tokenizer=model_path,
                     # config={'max_length': 1024}
                     )

gen_text = generator(text_inputs=your_text,
                     max_length=50,
                     do_sample=True,
                     temperature=0.7)[0]['generated_text']
print(gen_text)


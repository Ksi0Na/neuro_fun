# import torch
# import tensorflow as tf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gpt_2_simple as gpt2
import os
import requests

# model_name = "124M"
model_name = "774M"

model_dir = "gpt2\models\eng"
your_text = "The quick brown fox"

if not os.path.isdir(os.path.join(model_dir, model_name)):
    print(f"Downloading gpt-2 {model_name} model...")
    gpt2.download_gpt2(model_dir=model_dir, model_name=model_name)

# загрузка модели в TensorFlow сессию
session = gpt2.start_tf_sess()
gpt2.load_gpt2(session, model_name=model_name, model_dir=model_dir)

gen_text = gpt2.generate(sess=session,
                         model_name=model_name,
                         prefix=your_text,
                         length=50,
                         temperature=0.7,
                         top_p=0.5,
                         include_prefix=True,
                         model_dir=model_dir
                         # return_as_list=True
                         )[0]
print(gen_text)


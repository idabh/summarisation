import nltk
import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

timestr = time.strftime("%d-%H%M%S")
nltk.download('punkt')
model_checkpoint = "google/mt5-small"
metric = datasets.load_metric("rouge")

df = Dataset.from_pandas(pd.DataFrame({'text': ['Der er onsdag klokken 15 pressemøde om den aktuelle status for covid-19 i Danmark. Det oplyser Sundhedsstyrelsen i en pressemeddelelse. Ingen ministre er med til pressemødet, men blandt andre styrelsens direktør, Søren Brostrøm, deltager til det, der kaldes en pressebriefing. Det gør også faglig direktør i Statens Serum Institut Tyra Grove Krause og Lisbet Zilmer-Johns. Hun er direktør for Styrelsen for Forsyningssikkerhed. Det er den myndighed, der blandt andet har ansvar for testsystemet.']}))

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

######################## Evaluation ##################################
model = AutoModelForSeq2SeqLM.from_pretrained('./work/checkpoint-7500')
model.to('cpu')

batch_size = 4 

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    input_ids = inputs.input_ids.to("cpu")
    attention_mask = inputs.attention_mask.to("cpu")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = df.map(generate_summary, batched=True, batch_size=batch_size)
pred_str = results["pred"]

print(pred_str)

#from numpy import save
#np.save('./mt5' + timestr + '_preds.npy', pred_str)

import nltk
import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import time

timestr = time.strftime("%d-%H%M%S")
timestr = timestr + '_prefix'
nltk.download('punkt')
model_checkpoint = "google/mt5-small"
metric = datasets.load_metric("rouge")

train = Dataset.from_pandas(pd.read_csv("./danewsroom/train_d.csv", usecols=['text','summary','idx']))
test = Dataset.from_pandas(pd.read_csv("./danewsroom/test_d.csv", usecols=['text','summary','idx']))
val = Dataset.from_pandas(pd.read_csv("./danewsroom/val_d.csv", usecols=['text','summary','idx']))

dd = datasets.DatasetDict({"train":train,"validation":val,"test":test})
dd

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

######################## Evaluation ##################################
model = AutoModelForSeq2SeqLM.from_pretrained('./danewsroom/mt522-090648/checkpoint-7500')
model.to('cpu')
test_data = dd['test']

batch_size = 4  #
prefix = "summarize: "

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512 
    batch['text'] = [prefix + doc for doc in batch["text"]]
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    input_ids = inputs.input_ids.to("cpu")
    attention_mask = inputs.attention_mask.to("cpu")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_data.map(generate_summary, batched=True, batch_size=batch_size)

pred_str = results["pred"]
label_str = results["summary"]

rouge_output = metric.compute(predictions=pred_str, references=label_str)

from numpy import save
np.save('./mt5' + timestr + '_preds.npy', results)
np.save('./mt5' + timestr + '_test.npy', rouge_output)
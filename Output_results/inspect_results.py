import numpy as np

#--- @MAKE THIS A FUNCTION TO MAKE NEW PREDICTIONS BASED ON TEXT INPUT
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

#---


met = np.load("/work/summarisation/Output_results/mt5_metrics-50k.npy", allow_pickle=True)
met

res = np.load("/work/summarisation/Output_results/mt5_results-50k.npy", allow_pickle=True)
res[:20]

rou = np.load("/work/summarisation/Output_results/mt5_rouge-50k.npy", allow_pickle=True)
rou

#-----MT5 50K NROWS WITH 10 EPOCHS (run 16/12):
import numpy as np
import pandas as pd
met = np.load('mt5_50k_ep10_16_12_2021.npy', allow_pickle=True) #metrics
res = np.load('results_MT5_50k_ep10_16_12_2021.npy', allow_pickle=True) #results
rou = np.load('rouge_MT5_50k_ep10_16_12_2021.npy', allow_pickle=True) #rouge_output
#rouge: 0.54787443, 0.28235127, 0.35684455
res[:5]

#MT5 100K BLANDET
preds = np.load('/work/Summarization/mt522-090648_preds.npy', allow_pickle=True)
preds[:-5]

test = np.load('/work/Summarization/mt522-090648_test.npy', allow_pickle=True)
test

#idx 63148
df = pd.read_csv(r'/work/Summarization/danewsroom.csv', nrows = 63160)
df['summary'][63147]
df['text'][63147]
df['archive'][63147]
her = df.loc[df['Unnamed: 0'] == 63148]

#MAKE PREDS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained('/work/Summarization/mt522-090648/checkpoint-7500')

model.generate('det her er en test hurra!')



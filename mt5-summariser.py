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
timestr = timestr + "_abs_200k"
nltk.download('punkt')
model_checkpoint = "google/mt5-small"
metric = datasets.load_metric("rouge")

############################## Data ################################
#load through pandas
train = Dataset.from_pandas(pd.read_csv("abs_train_200k.csv", usecols=['text','summary','idx']))
test = Dataset.from_pandas(pd.read_csv("abs_test_200k.csv", usecols=['text','summary','idx']))
val = Dataset.from_pandas(pd.read_csv("abs_val_200k.csv", usecols=['text','summary','idx']))

# train = train.select(range(10000))
# val = val.select(range(1000))
# test = test.select(range(100))

#make the datasetdict
dd = datasets.DatasetDict({"train":train,"validation":val,"test":test})
dd

####################### Preprocessing #################################
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if model_checkpoint in ["google/mt5-small"]:
    prefix = "summarize: "
else:
    prefix = ""

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dd.map(preprocess_function, batched=True)

##################### Fine-tuning ############################
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 4
args = Seq2SeqTrainingArguments(
    output_dir = "./mt5" + timestr,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #weight_decay=0.01,
    #logging_steps=2000,  # set to 2000 for full training
    #save_steps=500,  # set to 500 for full training
    #eval_steps=7500,  # set to 7500 for full training
    #warmup_steps=3000,  # set to 3000 lsor full training
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    overwrite_output_dir= True,
    #fp16=True,
    load_best_model_at_end = True,
    metric_for_best_model='loss'
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value.mid.fmeasure for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    metrics={k: round(v, 4) for k, v in result.items()}
    return metrics

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

result=trainer.evaluate()
from numpy import save
save('./mt5' + timestr + '_train', result)  

######################## Evaluation ##################################
model.to('cuda')
test_data = dd['test']

prefix = "summarize: "

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    batch['text'] = [prefix + doc for doc in batch["text"]]
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

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

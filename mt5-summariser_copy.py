import nltk
import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import time



nltk.download('punkt')
model_checkpoint = "google/mt5-small"
metric = datasets.load_metric("rouge")

############################## Data ################################
#load through pandas and turn into Dataset format
#df = pd.read_csv(r'/work/NLP/danewsroom.csv', nrows = 10)
#df = pd.read_csv('/work/Summarization/danewsroom.csv', nrows = 10)
df = pd.read_csv('danewsroom.csv', nrows = 10)
df = df.rename(columns={'Unnamed: 0': 'idx'})
df_small = df[['text', 'summary', 'idx']]
data = Dataset.from_pandas(df_small)

#test train split
train_d, test_d = data.train_test_split(test_size=0.2).values() # , random_state = seed
#and validation
train_d, val_d = train_d.train_test_split(test_size=0.25).values()

#make the datasetdict
dd = datasets.DatasetDict({"train":train_d,"validation":val_d,"test":test_d})
dd

####################### Preprocessing #################################
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if model_checkpoint in ["google/mtf5-small"]:
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
timestr = time.strftime("%Y%m%d-%H%M%S")
batch_size = 4
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir = "./mt5" + timestr,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    overwrite_output_dir= True,
    #fp16=True,
    #push_to_hub=True,
    load_best_model_at_end = True
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
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    metrics={k: round(v, 4) for k, v in result.items()}
    np.save('mt5_metrics.npy', metrics) 
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

######################## Evaluation ##################################
test_data = dd['test']

batch_size = 16  # change to 64 for full evaluation

# map data correctly
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

results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["text"])

pred_str = results["pred"]
label_str = results["summary"]

rouge_output = metric.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

np.save('mt5_results.npy', results)
np.save('mt5_rouge.npy', rouge_output)

#ida trying:
from numpy import save
from numpy import savetxt
save('results_hallelujah.npy', results)
savetxt('rouge_heureka.csv', rouge_output)

results = np.load('mt5_results.npy', allow_pickle=True)
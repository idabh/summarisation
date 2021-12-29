import nltk
import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, EncoderDecoderModel, TrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass, field
from typing import Optional
import time

timestr = time.strftime("%d-%H%M%S")
nltk.download('punkt')
metric = datasets.load_metric("rouge")

############################## Data ################################
#load through pandas
train = Dataset.from_pandas(pd.read_csv("train_d.csv", usecols=['text','summary','idx']))
test = Dataset.from_pandas(pd.read_csv("test_d.csv", usecols=['text','summary','idx']))
val = Dataset.from_pandas(pd.read_csv("val_d.csv", usecols=['text','summary','idx']))

#make the datasetdict
dd = datasets.DatasetDict({"train":train,"validation":val,"test":test})
dd

####################### Preprocessing #################################
tokenizer = BertTokenizerFast.from_pretrained("Maltehb/danish-bert-botxo")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

batch_size=4  # change to 16 for full training
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

# only use 32 training examples for notebook - DELETE LINE FOR FULL TRAINING
train_data = dd['train']#.select(range(1500))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "summary", "idx"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
val_data = dd['validation']#.select(range(200))

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "summary", "idx"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

##################### Fine-tuning ############################
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("Maltehb/danish-bert-botxo", "Maltehb/danish-bert-botxo", tie_encoder_decoder=True)

# set special tokens
bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
bert2bert.config.eos_token_id = tokenizer.eos_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size #encoder
bert2bert.config.max_length = 128 #was 142?
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 2
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 3

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    f"bert" + timestr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #gradient_accumulation_steps=2,
    predict_with_generate=True,
    #evaluation_strategy = "steps",
    #save_strategy = "steps",
    do_train=True,
    do_eval=True,
    logging_steps=1000,  # set to 1000 for full training
    save_steps=500,  # set to 500 for full training
    eval_steps=500,  # set to 8000 for full training
    warmup_steps=2000,  # set to 2000 for full training
    #max_steps=16, # delete for full training
    num_train_epochs=1,
    overwrite_output_dir=True,
    save_total_limit=1,
    #load_best_model_at_end=True,
    #metric_for_best_model='rouge2',
    #generation_max_length=142
    #fp16=True, 
)

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

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
    )

trainer.train()

result=trainer.evaluate(max_length=128, num_beams=3)
from numpy import save
save('./bert' + timestr + '_train', result)  

######################## Evaluation ##################################
bert2bert.to("cuda")

test_data = dd['test']#.select(range(10))

#batch_size = 64  # change to 64 for full evaluation

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_data.map(generate_summary, batched=True, batch_size=batch_size)

pred_str = results["pred"]
label_str = results["summary"]

rouge_output = metric.compute(predictions=pred_str, references=label_str, use_stemmer=False)

from numpy import save
save('./bert' + timestr + '_preds', results)
save('./bert' + timestr + '_test', rouge_output)

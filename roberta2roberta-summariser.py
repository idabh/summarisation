import nltk
import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, EncoderDecoderModel, TrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
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
tokenizer = RobertaTokenizerFast.from_pretrained("flax-community/roberta-base-danish")

batch_size=4  # change to 16 for full training
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):                                                               
    # Tokenizer will automatically set [BOS] <text> [EOS]                                               
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)
                                                                                                        
    batch["input_ids"] = inputs.input_ids                                                               
    batch["attention_mask"] = inputs.attention_mask                                                     
    batch["decoder_input_ids"] = outputs.input_ids                                                      
    batch["labels"] = outputs.input_ids.copy()                                                          
    # mask loss for padding                                                                             
    batch["labels"] = [                                                                                 
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]                     
    batch["decoder_attention_mask"] = outputs.attention_mask                                                                              
                                                                                                         
    return batch  

# only use 32 training examples for notebook - DELETE LINE FOR FULL TRAINING
train_data = dd['train']#.select(range(100))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "summary"],
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
val_data = dd['validation']#.select(range(10))

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "summary"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

##################### Fine-tuning ############################
# set encoder decoder tying to True
roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("flax-community/roberta-base-danish", "flax-community/roberta-base-danish", tie_encoder_decoder=True)

# set special tokens
roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
roberta_shared.config.eos_token_id = tokenizer.eos_token_id

# sensible parameters for beam search
# set decoding params                               
roberta_shared.config.max_length = 128
roberta_shared.config.early_stopping = True
roberta_shared.config.no_repeat_ngram_size = 3
roberta_shared.config.length_penalty = 2.0
roberta_shared.config.num_beams = 4
roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size 

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
    f"roberta" + timestr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    #evaluation_strategy = "steps",
    #save_strategy = "steps",
    do_train=True,
    do_eval=True,
    logging_steps=2000,  # set to 2000 for full training
    save_steps=500,  # set to 500 for full training
    eval_steps=500,  # set to 7500 for full training
    warmup_steps=3000,  # set to 3000 for full training
    #max_steps=16, # delete for full training
    num_train_epochs=1, #uncomment for full training
    overwrite_output_dir=True,
    save_total_limit=1,
    #load_best_model_at_end=True,
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
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()

result=trainer.evaluate(max_length=128, num_beams=4)
from numpy import save
save('./roberta' + timestr + '_train', result)  

######################## Evaluation ##################################
roberta_shared.to("cuda")

test_data = dd['test']#.select(range(10))

#batch_size = 16  # change to 64 for full evaluation

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = roberta_shared.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_data.map(generate_summary, batched=True, batch_size=batch_size)

pred_str = results["pred"]
label_str = results["summary"]

rouge_output = metric.compute(predictions=pred_str, references=label_str, use_stemmer=False)

from numpy import save
save('./roberta' + timestr + '_preds', results)
save('./roberta' + timestr + '_test', rouge_output)
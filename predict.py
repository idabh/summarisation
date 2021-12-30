import nltk
nltk.download('punkt')
nltk.download('stopwords')
import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from tf_idf_summariser_scr import summarise as summarise_ex

#---FUNCTION
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
#---

#timestr = time.strftime("%d-%H%M%S")
model_checkpoint = "google/mt5-small"
metric = datasets.load_metric("rouge")

#df = Dataset.from_pandas(pd.DataFrame({'text': ['Der er onsdag klokken 15 pressemøde om den aktuelle status for covid-19 i Danmark. Det oplyser Sundhedsstyrelsen i en pressemeddelelse. Ingen ministre er med til pressemødet, men blandt andre styrelsens direktør, Søren Brostrøm, deltager til det, der kaldes en pressebriefing. Det gør også faglig direktør i Statens Serum Institut Tyra Grove Krause og Lisbet Zilmer-Johns. Hun er direktør for Styrelsen for Forsyningssikkerhed. Det er den myndighed, der blandt andet har ansvar for testsystemet.']}))

text = """
summarize: Årets næstsidste dag er startet med endnu en diset morgen mange steder.
Dog er sigtbarheden markant bedre end onsdag morgen. Kun enkelte steder i den sydlige del af landet kan der ligge nogle tætte tågebanker.
I løbet af dagen vil det fortsætte overskyet i en stor del af landet. Men særligt i Jylland kan man være heldig, at der kan kommer lidt opbrud i skydækket.
Derudover bliver det en meget mild dag. Allerede fra morgenstunden ligger temperaturen på mellem tre og syv graders varme. I dagens løb vil temperaturen stige op til ti graders varme.
På trods af de mange skyer, begynder torsdagen uden nævneværdigt nedbør.
Vi skal hen sidst på formiddagen, før der begynder at dukke spredte byger op i den sydvestlige del af landet.
Det bliver med skyet vejr i hele landet, men i den nordlige del af Jylland kan solen omkring middagstid titte lidt frem.
Spredte byger på en mild eftermiddag
Mens solen måske skinner lidt i det nordlige Jylland, vil bygerne, der gik i land i Sydvestjylland, efterhånden brede sig til resten af landet.
Først er det de vestlige og sydlige egne, der får spredte byger. Sidenhen er det også de østlige landsdele.
I løbet af eftermiddagen når temperaturen op mellem seks og ni graders varme. Køligst bliver det på Bornholm, mens det lokalt andre steder kan blive op til ti grader.
Vinden er dagen igennem let til jævn fra sydvest.
I aften vil det fortsat være overvejende skyet vejr.
I de østlige egne vil der stadig være spredte byger, der forventes at klinge af i aftenens løb. Det er fortsat mildt med syv til ni grader på termometret.
"""

#the manually pasted text from above:
#df = Dataset.from_pandas(pd.DataFrame({'text': [str(text)]}))

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

######################## Evaluation ##################################
#-RANDOM MT5 (7500 is best???):
model_ran = AutoModelForSeq2SeqLM.from_pretrained('/work/Summarization/mt522-090648/checkpoint-7500')
#-ABSTRACTIVE MT5(15000 is best?):
model_abs = AutoModelForSeq2SeqLM.from_pretrained('/work/Summarization/mt527-140701_abs/checkpoint-15000')
#-MIXED MT5:
model_mix = AutoModelForSeq2SeqLM.from_pretrained('/work/Summarization/mt528-132950_mix/checkpoint-15000')
#-EXTRACTIVE MT5 (15000 is best?):
model_ex = AutoModelForSeq2SeqLM.from_pretrained('/work/Summarization/mt526-142648_ex/checkpoint-15000')

models = [model_ran, model_abs, model_mix, model_ex]


#articles from the test sets:
abs_test = Dataset.from_pandas(pd.read_csv("/work/Summarization/abs_test.csv"))
mix_test = Dataset.from_pandas(pd.read_csv("/work/Summarization/mix_test.csv"))
ex_test = Dataset.from_pandas(pd.read_csv("/work/Summarization/ex_test.csv"))
#add 'summarize: ' PREFIX since the models were trained on that!:
abs_test = abs_test.map(lambda example: {'text': 'summarize: ' + example['text']})
mix_test = mix_test.map(lambda example: {'text': 'summarize: ' + example['text']})
ex_test = ex_test.map(lambda example: {'text': 'summarize: ' + example['text']})


#CHANGE ARTICLES HERE
articles = mix_test[7003:7023]

#####GENERATE SUMMARIES#####
for article in articles['text']:
    df = Dataset.from_pandas(pd.DataFrame({'text': [str(article)]}))
    print(df['text']) #print the article
    for model in models:
        model.to('cpu')
        results = df.map(generate_summary, batched=True, batch_size=batch_size)
        pred_str = results["pred"]
        print(pred_str)

        #from numpy import save
        #np.save('./mt5' + timestr + '_preds.npy', pred_str)
    
    #Extractive TF-IDF summariser:
    print(summarise_ex(df['text'][0], 3)) #if time: remove 'summarize: ' prefix from this version
    print("---NEXT ARTICLE---")

#ORDER OF PRINTING: RANDOM, ABSTRACTIVE, MIXED, EXTRACTIVE, TF-IDF
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

#df = Dataset.from_pandas(pd.DataFrame({'text': ['Der er onsdag klokken 15 pressemøde om den aktuelle status for covid-19 i Danmark. Det oplyser Sundhedsstyrelsen i en pressemeddelelse. Ingen ministre er med til pressemødet, men blandt andre styrelsens direktør, Søren Brostrøm, deltager til det, der kaldes en pressebriefing. Det gør også faglig direktør i Statens Serum Institut Tyra Grove Krause og Lisbet Zilmer-Johns. Hun er direktør for Styrelsen for Forsyningssikkerhed. Det er den myndighed, der blandt andet har ansvar for testsystemet.']}))

text = """
summarize: Magnus Carlsen holdt sig ikke tilbage i sin kritik. Den norske skakstjerne var rasende.
- Det er en idiotisk regel. Enten skal alle med samme pointsum spille omspil, eller også skal ingen, siger Magnus Carlsen til NRK.
Han var på førstepladsen, men på finaledagen ved VM i hurtigskak i Polen gik det hele galt for den forsvarende mester på grund af en særlig regel.
Min far og jeg skulle have været vagthunde over for FIDE. Hvis ikke man er det, fucker de op hver gang.
Magnus Carlsen, Nodirbek Abdusattorov, Ian Nepomniachtchi og Fabiano Caruana havde alle lige mange point. Derfor skulle to af dem i omspil.
Her foreskriver reglerne, at det er de to, som har indsamlet den højeste score efter Sonneborn-Berger-regnemetoden, der går i omspil.
Sonneborn-Berger-regnemetoden udregner en score på baggrund af niveauet af de spillere, man har besejret og spillet uafgjort mod. Det var her, Abdusattorov og Nepomniachtchi var de to bedste, mens Carlsen var tredjebedst og Caruana fjerdebedst.
Efter en udregning blev det bekendtgjort, at Magnus Carlsen var tredjebedst og derfor missede VM-finalen. Det er en ekstrem sur situation, fortæller nordmanden.
- De fleste mennesker vil mene, at de ikke er fair. Simpelthen. Reglerne skal ændres. Omspil er et skridt i den rigtige retning, men det er vanskeligt at se andet end logistiske årsager til, at det kun er to som skal i omspil.
- Det virker amatøragigt ved et VM, mener Magnus Carlsen.
Den norske verdensmester i skak bliver støttet af sine kollegaer Hikaru Nakamura fra USA og russiske Sergej Karjakin. På Twitter kalder amerikaneren regelsættet for grinagtigt.
Ifølge Magnus Carlsen lider retfærdigheden et alvorligt nederlag, når reglen om omspil afvikles som ved VM i hurtigskak. Det er helt korrekt at benytte sig af den ved pointlighed, men så skal alle have muligheden for at spille om, understreger han.
Magnus Carlsen påtager sig dog skylden for, at han missede chancen for at vinde VM-guldet.
- Min far og jeg skulle have været vagthund over for FIDE (Det internationale skakforbund, red.). Hvis ikke man er det, fucker de op hver gang. Sådan er det åbenbart. Jeg tror, at de fleste mener, det er uretfærdigt, siger Magnus Carlsen.
Ifølge FIDE er det på grund af TV-rettigheder svært at ændre reglerne for omspil, så de omfatter mere end to spillere. Det fortæller David Llada, forbundets kommunikationschef.
- Ideen med omspil mellem to spillere virker god. Men vi tager tilbagemeldingen til efterretning. Der findes dog mere passende formuleringer til at udtrykke sin mening, men vi forstår, at enkelte er frustrerede, siger han.
Magnus Carlsen har vundet VM i hurtigskak tre gange. Han missede dog muligheden for at genvinde titlen, som han nappede i 2019, da det senest blev afholdt.
- Jeg tager i hvert fald en medalje med hjem, og det er bedre end ingenting. Det var ikke indlysende, at jeg ville klare det efter min dårlige start i dag, fortæller han.
Nordmanden har dog allerede mulighed for at samle brikkerne onsdag. Der begynder VM i lynskak, som også afholdes i Polen og varer til og med den 30. december.
"""

df = Dataset.from_pandas(pd.DataFrame({'text': [str(text)]}))

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

for model in models:

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

from tf_idf_summariser_scr import summarise_danewsroom
import pandas as pd

df = pd.read_csv(r'../danewsroom.csv', nrows = 100)
# Run on the first n samples
output = summarise_danewsroom(df, 15)
output[0]
output[1]
output[2]
df["text"]
output[2]

# Run on the more extractive samples
extractive = df[df["density"] > 8.1875] 
len(df[df["density"] > 8.1875])
extractive = pd.DataFrame.reset_index(extractive)
output_ex = summarise_danewsroom(extractive, 15)
output_ex[0]
output_ex[1]
output_ex[2]

#-- look at scores --
output[0] # make a table with the scores?

#---look at results of the texts summarisation ---
checking = pd.DataFrame(list(zip(output[1], output[2])), columns =['Human', 'Generated'])
#pd.set_option('display.max_colwidth', 50)
checking
output[1][0] #human
output[2][0] #extracted
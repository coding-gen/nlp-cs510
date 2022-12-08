#! /usr/bin/python3

import pandas as pd
import numpy as np


# expun_pilot = pd.read_json("expunations/data/expunations_annotated_pilot_100.json").rename(columns={'ID': 'pun_id'}).set_index('pun_id')
expun = pd.read_json("expunations/data/expunations_annotated_full.json").rename(columns={'ID': 'pun_id'}).set_index('pun_id')

def get_puns(path):
  import xml.etree.ElementTree as ETree
  xmldata = path
  prstree = ETree.parse(xmldata)
  root = prstree.getroot()

  puns = {}
  i=0
  for sequence in root.iter('text'):
    id = sequence.attrib.get('id')
    pun = []
    for sequence_word in sequence.iter():
      if sequence_word.text != '\n    ':
        pun.append(sequence_word.text)
    puns[i] = {'id': id, 'pun': pun}
    i += 1

  # puns: {'row_0': {'id': 't_1', 'pun': ['When', 'the', ... ]}}
  return pd.DataFrame.from_dict(puns, orient='index')

df = get_puns("semeval2017_task7/data/trial/subtask1-heterographic-trial.xml").rename(columns={'id': 'pun_id'}).set_index('pun_id')
heterographic = get_puns("semeval2017_task7/data/test/subtask1-heterographic-test.xml").rename(columns={'id': 'pun_id'}).set_index('pun_id')
homographic = get_puns("semeval2017_task7/data/test/subtask1-homographic-test.xml").rename(columns={'id': 'pun_id'}).set_index('pun_id')
hetero_labels = pd.read_csv("semeval2017_task7/data/test/subtask1-heterographic-test.gold", names = ['id', 'label'], delim_whitespace=True).rename(columns={'id': 'pun_id'}).set_index('pun_id')
homograph_labels = pd.read_csv("semeval2017_task7/data/test/subtask1-homographic-test.gold", names = ['id', 'label'], delim_whitespace=True).rename(columns={'id': 'pun_id'}).set_index('pun_id')

heterographic['label'] = hetero_labels['label']
homographic['label'] = homograph_labels['label']
labelled_puns = pd.concat([heterographic, homographic])
pd.reset_option('display.max_colwidth', None)
dataset = expun.join(labelled_puns, on='pun_id', how='left')

del labelled_puns
del heterographic
del homographic 
del homograph_labels
del hetero_labels


dataset['Joke keywords'] = dataset['Joke keywords'].apply(np.hstack) #flatten
dataset['Joke keywords'] = dataset['Joke keywords'].apply(str) #bc text input must be of type str
dataset['Joke keywords'] = dataset['Joke keywords'].apply(str.lower) 
dataset['pun'] = dataset['pun'].apply(str) 
dataset['pun'] = dataset['pun'].apply(lambda x: ''.join(x)) 
# print("a_process")
# print(dataset['Joke keywords'].head())
# print(dataset['pun'].head())


#train/test dataset
from sklearn.model_selection import train_test_split
from pandas.core.common import flatten

#need source & target
dataset = dataset[dataset['label'] ==1] #getting only the positive datapoints
limited_dataset = dataset[['Joke keywords', 'pun']] #only taking the columns we need

#need to rename columns for t5 training
limited_dataset.columns = ["source_text", "target_text"]
#prepend the prefix:
limited_dataset['source_text'] = "generate sentence: " + limited_dataset['source_text']

print(limited_dataset.head())

train, test = train_test_split(limited_dataset, test_size=0.3, random_state=0)
#split in half to get both test and eval:
test, eval = train_test_split(test, test_size=0.5, random_state=0)



# Training T5

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained("t5","t5-base")

# train (from simplet5 docs)
# but first we need the data in the right format! see block above
model.train(train_df=train, # pandas dataframe with 2 columns: source_text & target_text
            eval_df=eval, # pandas dataframe with 2 columns: source_text & target_text
            source_max_token_len = 512, 
            target_max_token_len = 128,
            batch_size = 8,
            max_epochs = 5,
            use_gpu = False,
            outputdir = "outputs",
            early_stopping_patience_epochs = 0,
            precision = 32
            )

"""
for i in test.index:
        candidateSentences1 = model.predict(test.at[i, "source_text"])
        print(test.at[i, 'source_text'])
        print("Candidate sentences : ", candidateSentences1)
""""
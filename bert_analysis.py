import argparse
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import wikipedia

import bert
from bert_utils import cos_similarity


MODEL_LS = ['bert-base-uncased']
SIM_FNAMES = ['data/WordSim-353.csv', 'data/SimLex-999.csv', 'data/MEN.csv']


# make an argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", "-m", metavar='MODEL', choices=MODEL_LS, default=MODEL_LS[0],
    help='loading local model among: ' + ' | '.join(MODEL_LS)\
    + ' (default: ' + MODEL_LS[0]
)

args = parser.parse_args()
MODEL_DIR_PATH = 'data/bert_model/'
MODEL_NAME = MODEL_DIR_PATH + args.model


def generate_w2vs_from_csv(wiki=False):
    # wiki: if True, using wikipedia's first sentence to generate word vector
    
    bert_model = bert.Bert(MODEL_NAME)
    print("Bert Class is created")

    # similarity dataset
    for sim_fname in tqdm(SIM_FNAMES):
        sim_df = pd.read_csv(sim_fname)
        bert_cos = [None] * len(sim_df) # initialize bert_cos
        sim_df['bert_cos'] = bert_cos
    
        for index, row in tqdm(sim_df.iterrows()):
            if wiki:
                try:
                    raw_txt1 = wikipedia.summary(row["word1"], sentences=1)
                    raw_txt2 = wikipedia.summary(row["word2"], sentences=1)
                    bert_vec1 = bert_model.get_bert_w2v(raw_txt1, sentence=True)
                    bert_vec2 = bert_model.get_bert_w2v(raw_txt2, sentence=True)
                except:
                    # Disambiguation error
                    sim_df.at[index, 'bert_cos'] = np.nan
                    continue
            else:
                bert_vec1 = bert_model.get_bert_w2v(row["word1"])
                bert_vec2 = bert_model.get_bert_w2v(row["word2"])
            cos = cos_similarity(bert_vec1, bert_vec2)  # cos similarity
            sim_df.at[index, 'bert_cos'] = cos
        save_fname = 'data/bert_model/' +\
            os.path.basename(sim_fname).replace('.csv', '_bert.csv')
        sim_df.to_csv(save_fname)
        # print("{} is generated".format(save_fname))


def analyze():
    directory = "data/bert_model/wiki/"
    
    men = pd.read_csv(directory + "MEN_bert.csv")
    men.dropna(inplace=True)
    simlex = pd.read_csv(directory + "SimLex-999_bert.csv")
    simlex.dropna(inplace=True)
    wordsim = pd.read_csv(directory + "WordSim-353_bert.csv")
    wordsim.dropna(inplace=True)
    
    datasets = [
        {
            "df": men,
            "name": "MEN"
        },
        {
            "df": simlex,
            "name": "SimLex"
        },
        {
            "df": wordsim,
            "name": "WordSim"
        }
    ]

    for dataset in datasets:
        print(
            dataset["name"],
            pearsonr(dataset["df"]["result"], dataset["df"]["bert_cos"])
        )

    
        
if __name__ == "__main__":
    # generate_w2vs_from_csv(wiki=True)
    analyze()

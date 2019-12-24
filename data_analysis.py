import pandas as pd
from scipy.stats import pearsonr

from word2vec import Word2Vec

if __name__ == "__main__":
    modelname = "glove-wiki-gigaword-100"
    wordsim = pd.read_csv("data/similarity/{}-WordSim.csv".format(modelname))
    simlex = pd.read_csv("data/similarity/{}-SimLex.csv".format(modelname))
    men = pd.read_csv("data/similarity/{}-MEN.csv".format(modelname))
    w2v = Word2Vec(modelname=modelname)

    datasets = [
        {
            "df": wordsim,
            "name": "WordSim"
        }, 
        {
            "df": simlex,
            "name": "SimLex"
        },
        {
            "df": men,
            "name": "MEN"
        }
    ]
    for dataset in datasets:
        print(
            dataset["name"],
            pearsonr(dataset["df"][dataset["name"]], dataset["df"]["cos"])
        )


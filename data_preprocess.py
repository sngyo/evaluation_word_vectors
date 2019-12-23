import pandas as pd

from word2vec import Word2Vec


if __name__ == "__main__":
    # WordSimとSimLexの共通単語対を作る
    wordsim = pd.read_csv("data/WordSim-353.csv")
    simlex = pd.read_csv("data/SimLex-999.csv")
    # target_dict = {
    #     "word1": [],
    #     "word2": [],
    #     "WordSim": [],
    #     "SimLex": []
    # }
    # for index, row in wordsim.iterrows():
    #     tmpdf = simlex.query(
    #         "word1 == '{}' or word2 == '{}'".format(row["Word 1"], row["Word 1"])
    #     )
    #     if not tmpdf.empty:
    #         tmpdf = tmpdf.query(
    #             "word1 == '{}' or word2 == '{}'".format(row["Word 2"], row["Word 2"])
    #         )
    #         if not tmpdf.empty:
    #             # print(row["Word 1"], row["Word 2"])
    #             # print(tmpdf)
    #             target_dict["word1"].append(row["Word 1"])
    #             target_dict["word2"].append(row["Word 2"])
    #             target_dict["WordSim"].append(row["Human (mean)"])
    #             target_dict["SimLex"].append(tmpdf.iloc[0, 3])
    # result_df = pd.DataFrame.from_dict(target_dict)
    # print(result_df)
    # result_df.to_csv("data/commonpairs.csv", index=False)

    # MENとcommonの共通単語確認
    men = pd.read_csv("data/MEN.csv")
    common = pd.read_csv("data/commonpairs.csv")
    # target_dict = {
    #     "word1": [],
    #     "word2": [],
    #     "WordSim/10": [],
    #     "SimLex/10": [],
    #     "MEN/50": []
    # }
    # for index, row in simlex.iterrows():
    #     tmpdf = men.query(
    #         "word1 == '{}' or word2 == '{}'".format(row["word1"], row["word1"])
    #     )
    #     if not tmpdf.empty:
    #         tmpdf = tmpdf.query(
    #             "word1 == '{}' or word2 == '{}'".format(row["word2"], row["word2"])
    #         )
    #         if not tmpdf.empty:
    #             print(row["word1"], row["word2"])
    #             print(tmpdf)

    # 各データセットとword2vecの比較
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
    modelname = "word2vec-google-news-300"
    w2v = Word2Vec(modelname=modelname)
    for dataset in datasets:
        print(dataset["name"])
        target_dict = {
            "word1": [],
            "word2": [],
            dataset["name"]: [],
            "cos": []
        }
        for _, row in dataset["df"].iterrows():
            cos = w2v.get_cosine_similarity(row["word1"], row["word2"])
            if cos:
                target_dict["word1"].append(row["word1"])
                target_dict["word2"].append(row["word2"])
                target_dict[dataset["name"]].append(row["result"])
                target_dict["cos"].append(cos)
        result_df = pd.DataFrame.from_dict(target_dict)
        result_df.to_csv(
            "data/similarity/w2v-{}.csv".format(dataset["name"]),
            index=False
        )
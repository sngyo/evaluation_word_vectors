import pandas as pd


if __name__ == "__main__":
    # WordSimとSimLexの共通単語対を作る
    wordsim = pd.read_csv("data/WordSim-353.csv")
    simlex = pd.read_csv("data/SimLex-999.csv")
    target_dict = {
        "word1": [],
        "word2": [],
        "WordSim": [],
        "SimLex": []
    }
    for index, row in wordsim.iterrows():
        tmpdf = simlex.query(
            "word1 == '{}' or word2 == '{}'".format(row["Word 1"], row["Word 1"])
        )
        if not tmpdf.empty:
            tmpdf = tmpdf.query(
                "word1 == '{}' or word2 == '{}'".format(row["Word 2"], row["Word 2"])
            )
            if not tmpdf.empty:
                # print(row["Word 1"], row["Word 2"])
                # print(tmpdf)
                target_dict["word1"].append(row["Word 1"])
                target_dict["word2"].append(row["Word 2"])
                target_dict["WordSim"].append(row["Human (mean)"])
                target_dict["SimLex"].append(tmpdf.iloc[0, 3])
    result_df = pd.DataFrame.from_dict(target_dict)
    print(result_df)
    result_df.to_csv("data/commonpairs.csv", index=False)
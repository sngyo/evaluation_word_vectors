# evaluation_word_vectors
evaluation framework of word vectors

## Purpose
新しいワードベクトルを準備した時に，人間の感覚との近さを簡易的に評価したい
- step1．ワードベクトルデータを作る
- step2．提案するフレームワークに入れる
- step3．人の感覚との近さを数値で獲得する．


## Comment
### word2vec.py
Word2Vecクラスが入っている．
[Gensim-data](https://github.com/RaRe-Technologies/gensim-data)のモデルは全て使用可能.  
(デフォルトは `glove-wiki-gigaword-100`)  
初回読み込み時にデータをダウンロードし， pickle形式で data/ に保存する．

### data_preprocess.py & data_analysis.py
`data_preprocess.py`: 類似語データセットをcsvに統一形式でまとめる
`data_analysis.py`: 各モデルによって獲得された単語対の分散表現間のcos類似度と類似語データセットの類似度の分析実行スクリプト．

### bert.py & bert_utils.py & bert_analysis.py
PyTorchで公開されている学習済みBERTモデルを利用．
`bert.py`: BERTクラス
`bert_utils.py`: cos類似度関数などのツールを保管
`bert_analysis.py`: BERTクラスを実際に呼び出して単語の分散表現を獲得・分析実行スクリプト


## Future Works
- [ ] wikipediaのsummaryのsentence vectorを賢くする
  - [ ] tf-idfをかけることでキーワード抽出
  - [ ] sentence vector用にfine-tuning([CLS]タグの部分がsentence vectorに相当するはず)
- [ ] TOEFL synonym question datasetの精度でも評価できる
- [ ] ピアソンの相関係数以外の手法で二つの分布の近さを調べる手法が欲しい．
- [ ] usabilityの向上
  - [ ] コメントの追加
  - [ ] サンプルスクリプトの作成


## References
- Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin, "Placing Search in Context: The Concept Revisited", ACM Transactions on Information Systems, 20(1):116-131, January 2002
- Felix Hill, Roi Reichart and Anna Korhonen, “SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation”, Computational Linguistics. 2014
- Elia Bruni, Nam Khanh Tran, and Marco Baroni. “Multimodal distributional semantics”, J. Artif. Int. Res. 49, 1 (January 2014), 1-47, 2014
- Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space”, arXiv preprint arXiv:1301.3781
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning, “Glove: Global vectors for word representation”, Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, 2014
- Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov, “Enriching Word Vectors with Subword Information”, arXiv preprint arXiv: 1607.04606
- Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer,  “Deep contextualized word representations”, NAACL, 2018a
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin, “Attention is all you need”, Advances in Neural Information Processing Systems, pages 6000–6010, 2017
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, arXiv preprint arXiv: 1810.04805
- Oren Melamud, David McClosky, Siddharth Patwardhan, Mohit Bansal, “The Role of Context Types and Dimensionality in Learning Word Embeddings”, NAACL, 2016
- Tobias Schnabel, Igor Labutov, David Mimno, Thorsten Joachims, “Evaluation methods for unsupervised word embeddings”, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 298-307, September 2015
- Amir Bakarov, “A Survey of Word Embeddings Evaluation Methods”, arXiv preprint arXiv 1801.09536
- Manaal Faruqui, Yulia Tsvetkov, Pushpendre Rastogi, Chris Dyer, “Problems With Evaluation of Word Embeddings Using Word Similarity Tasks”, arXiv preprint arXiv 1605.02276
- Yulia Tsvetkov, Manaal Faruqui, Chris Dyer, “Correlation-based Intrinsic Evaluation of Word Vector Representations”, arXiv preprint arXiv 1606.06710

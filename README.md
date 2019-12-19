# evaluation_word_vectors
evaluation framework of word vectors

## Word2vec 自体の精度を評価したい (NLPの課題レポート)
- 期限 : 2020/01/19 (sun) 23:50
- 現在 Scain が抱える問題点の一つが、単語の分散表現の精度に大きく依存している点
  - 分散表現自体の精度を簡単に計測できるようにしておくと便利
  
## Purpose
新しいワードベクトルを準備した時に，簡単に評価できるようなフレームワークを作る
- step1．ワードベクトルデータを作る
- step2．提案するフレームワークに入れる
- step3．既存のものより精度がよくなっているかの指標になれば嬉しい




## TODO
- [x] 先生へ連絡、返事待ち 
    > 斎藤です。システムの複数共同作成は可ですが、きちんと自分はどの分担
    だったかを明記してください。
    あと、やはり他でやった作業をそのまま転用するのはあまりにレポートと
    して手抜き感があるので、今回のレポート用にオリジナル作業を入れるよ
    うにしてください。(どの部分がオリジナルかを明記して)
- [x] githubかなんかで管理？
- [ ] Survey
- [ ] 里形との分担
- [ ] データベースの検索 (word2vec，評価用データセット双方
- [ ] フレームワークの骨子部分の作成


## 手法
- ~~外省的評価：タスク先の(感情分析とか)性能によって評価する手法~~  ← こっちはやらない
- 内省的評価：単語分散表現の性能を直接評価する方法。たとえば、人間が判断した単語類似度と分散表現による単語類似度の相関によって性能を評価。要するに、人間の評価に近い評価をできるようになれば良い。

## 使えそうなもの
評価用データセットとして [WordSim353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)や[SimLex999](https://fh295.github.io//simlex.html)。

## Idea
- 大きいスコープでざっくり見るのと、Domain Specific なものを別々に検証できるといいかも？ by Fukuchi-san

- まずは，コサイン類似度を利用？  
    類似語データセット(WordSim353やSimLex999)を利用  
    word2vec, Glove, (BERT), などの各手法のコサイン類似度を比較



### @sngyo's Ideas
- 大きいスコープでざっくり見るのと、Domain Specific なものを別々に検証できるといいかも？
- Word to Sentence Vector Space で作成する word embedding の精度検証にも役立ちそう、というかそれで検証していい精度を出せるものが作れたら論文にできない？
- @shoya 提案の数値を含めた Embedding も手法を考えてみたい。

## 結果

## References
- [単語分散表現の最適な次元数を決めるための指針](https://qiita.com/Hironsan/items/01fd880f1522e2025a78)
- [The Role of Context Types and Dimensionality in Learning Word Embeddings](https://arxiv.org/abs/1601.00893)
- [How to evaluate word embeddings](https://www.quora.com/How-do-I-evaluate-word-embeddings)




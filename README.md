fg# evaluation_word_vectors
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
- [x] 共同レポート可能かチェック
  - > 現在，二人で共同作業をしてそれぞれ異なるタイプのデータセットを用意
  するなどして検証の準備を行なっています．レポートに関しても共同で一つの
  ものを作成し(それぞれ提出する)，その中でどの部分を担当している，
  などを記載すればよろしいのでしょうか？
  - > はい、役割分担を明確に記述してください。
  
- [x] github化
- [ ] ~~gensim以外でword2vecのpretrained modelで使えそうなもの探す~~
- [x] Survey
- [x] 里形との分担
- [ ] 評価用データセット探す
  - [x] WordSim353, SimLex999のLoading Script作成 (WIP @takasago)
  - syntactic性をはかるデータセットも欲しい
- [ ] 評価の手法考えて実装
  - [ ] とりあえず cos類似度で検証
  - [ ] 他の手法も試せたら比較検証できる
- [ ] レポート


## Methods
- ~~外省的評価：タスク先の(感情分析とか)性能によって評価する手法~~  ← こっちはやらない
- 内省的評価：単語分散表現の性能を直接評価する方法。たとえば、人間が判断した単語類似度と分散表現による単語類似度の相関によって性能を評価。要するに、人間の評価に近い評価をできるようになれば良い。

## 使えそうなもの
評価用データセットとして [WordSim353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)や[SimLex999](https://fh295.github.io//simlex.html)。

## Ideas
- 大きいスコープでざっくり見るのと、Domain Specific なものを別々に検証できるといいかも？ by Fukuchi-san

- まずは，コサイン類似度を利用？  
    類似語データセット(WordSim353やSimLex999)を利用  
    word2vec, Glove, (BERT), などの各手法のコサイン類似度を比較


### @sngyo's Ideas
- 大きいスコープでざっくり見るのと、Domain Specific なものを別々に検証できるといいかも？
  - @shoya's Wine Embeddings
- Word to Sentence Vector Space で作成する word embedding の精度検証にも役立ちそう、というかそれで検証していい精度を出せるものが作れたら論文にできない？
  - 作ってみる？
- ちゃんとSurveyしないとわからないけど，semanticな判定とsyntacticな判定も行えそう．
  - fast <-> faster  (syntactic)
  - fast <-> rapid  (semantic)

- BERT で作成した word embedding が word2vec とかとどんなふうに違うのか定量的に評価できたらいいんじゃない？
  - 流れとしては，
  1. 研究室でのSCAINプロジェクト
  2. 賢いword embedding が欲しい
  3. 比較する方法があると嬉しいと思って作った
  4. 最近NLPで流行のBERTを使用する場合，どんな違いが生まれるか？


## Usage
[Gensim-data](https://github.com/RaRe-Technologies/gensim-data)のモデルは全て使用可能.  
初回読み込み時にデータをダウンロードし， pickle形式で data/ に保存する．  
デフォルトは `glove-wiki-gigaword-100`
```bash
python3 word2vec.py -m glove-wiki-gigaword-100

test word label: apple
('microsoft', 0.7449405789375305)
('ibm', 0.6821643710136414)
('intel', 0.6778088212013245)
('software', 0.6775422096252441)
('dell', 0.6741442680358887)
('pc', 0.6678153276443481)
('macintosh', 0.66175377368927)
('iphone', 0.6595611572265625)
('ipod', 0.6534676551818848)
('hewlett', 0.6516579985618591)
```

jupyter notebookをcommitする際には，Cell -> All Output -> Clear で余計なものを削除してから行うこと
jupter labの場合は Edit -> Clear All Output


## Results

## References
- [単語分散表現の最適な次元数を決めるための指針](https://qiita.com/Hironsan/items/01fd880f1522e2025a78)
- [The Role of Context Types and Dimensionality in Learning Word Embeddings](https://arxiv.org/abs/1601.00893)
- [How to evaluate word embeddings](https://www.quora.com/How-do-I-evaluate-word-embeddings)

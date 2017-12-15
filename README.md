# ViZDoom_DoubleDQN

![basic.png](https://github.com/YutaroKikuchi/ViZDoom_DoubleDQN/blob/master/images/basic.png)

Doomエンジンで，小さい部屋にいるモンスターを倒すエージェントを学習します．
Double DQNというアルゴリズムを使いました．

dockerを使っているので，学習だけならどの環境でもいけるはずです．
しかし，ゲーム画面をウィンドウ表示させるのはUbuntuでのみ保証されています．

##学習の様子

右が1学習ステップあたりに得られる報酬，左がネットワークの損失
![graph.png](https://github.com/YutaroKikuchi/ViZDoom_DoubleDQN/blob/master/images/graph.png)

学習過程のGIFとかを作ります．(そのうち)


#＃ 使い方

docker実行用のdocker イメージをビルドします．build.shがビルド用スクリプトです．
```
$ ./build.sh
```

イメージからdockerコンテナを起動します．run.shが起動用スクリプトです．
```
$ ./run.sh
```
起動後にコンテナ内のbashに飛ばされます．

学習を始める場合は，train_test.pyを実行してください．1から学習し，モデルを作り直します．
```
$ python3 train_test.py
```

すでに学習済みのモデルを動かすだけなら，demo.pyで可能です．第２引数に./models/0009.ckptをつけてください．
```
$ python3 demo.py ./model/0009.ckpt
```
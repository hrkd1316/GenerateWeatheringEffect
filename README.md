# GenerateWeatheringEffect

## 目的
オブジェクトの任意の領域に風化効果を生成する.

## 先行研究
["Single Image Weathering via Exemplar Propagation"](http://iizuka.cs.tsukuba.ac.jp/projects/weathering/weathering_eng.html)
既に風化領域のある画像において，風化の再現をしている．

## 提案手法
先行研究にある風化度マップをユーザが指定した領域に疑似的に再現し，風化効果を生成する．
入力は参照画像と対象画像の2枚．出力は対象画像に風化効果を生成したもの．

参照画像からは風化効果を生成するのに必要な情報を取得する．
対象画像にはユーザがブラシツールで風化させたい領域を入力する．
これらを組み合わせて出力画像を得る．

![overview](https://user-images.githubusercontent.com/48472692/72405792-6dd09200-379d-11ea-8489-b44a29a094db.png)

## 結果
参照画像

![ref_img1](https://user-images.githubusercontent.com/48472692/68087556-8e759500-fe9a-11e9-88d5-53f55ebfe08f.jpg)

対象画像，ユーザ入力，出力画像

![tgt_img1_output](https://user-images.githubusercontent.com/48472692/68088096-f0d09480-fe9e-11e9-9a44-32e2a325ace0.png)

## 課題
- テクスチャ生成の際の特別なコストの実装
- ユーザが使用することを意識したUI

# Object Detection (物体検出)  
This document is also available in [English](https://github.com/yusuke-1105/Object_Detection/blob/master/README_EN.md).   
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![I like Docker](https://img.shields.io/badge/I%20like-Docker%20-blue)]()
[![Python](https://img.shields.io/badge/Python-%20-red)]()  


## このプログラムについて  

これは顔認証、および物体認識を行うプログラムです。  
[OpenVINO™ Toolkit](https://01.org/openvinotoolkit)と[YOLO v3](https://pjreddie.com/darknet/yolo/)を使用しています。  
OpenVINOは画像処理やディープラーニングを用いた推論を行うことができるものです。詳しくは[公式サイト](https://www.intel.co.jp/content/www/jp/ja/internet-of-things/solution-briefs/openvino-toolkit-product-brief.html)を見てください。   
OpenVINOは[software.intel.com](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)でダウンロード可能です。  
ですが、少しややこしいので、下の[このプログラムの実行方法](#このプログラムの実行方法)に簡単な方法を書いています。

このレポジトリのプログラム部分は以下のような構成となっています。  

|フォルダ名|内容|
|:---:|:---:|
|`Models`|モデルやラベルリスト|
|`Post_Image`|処理された後の写真|
|`Pre_Image`|処理される前の写真|
|`main1.py`|顔認識をするプログラム|
|`main2.py`|物体認識をするプログラム|
|`Docker`|Dockerに関するファイル|

  
## このプログラムの実行方法  
### 実行方法1  
#### [Intel DevCloud for the Edge](https://devcloud.intel.com/edge/)  
これはIntelさんが用意したクラウドサービスです。あらかじめOpenVINOがインストールされており、自前のPCで環境を構築しなくても、OpenVINOを使用した画像処理や推論実行が可能となっています。  
はじめに上記の公式サイトにアクセスしてください。  

下の図が手順です。  
ここで④について、登録できていない方は登録してください。登録が済むと以下のようなメールが来るので、赤い丸で囲われているところのURLをコピーして、それを「submit」の上の記入欄に入力してください。  
⑤では「Use JupyterLab」の<b>チェック欄にチェックを入れてください</b>。  
⑧まで進んでください。  

<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/pic1.png">
</div>  

<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/pic2.png">
</div>  

ターミナルに以下のように入力してください。(コピペ→Enter、returnで大丈夫です)  
```
git clone https://github.com/yusuke-1105/Object_Detection
```  
すると以下のようにフォルダの中に「Object_Detection」が表示され、それをクリックすると、内容を見ることができます。  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/pic3.png">
</div>   

そして、以下のコマンドを入力してください。  
```
cd Object_Detection
wget https://www.dropbox.com/s/6dlv9448ssy8lki/Archive.zip?dl=0 -O Models/Archive.zip
unzip Models/Archive.zip -d Models/
```
ここで`Pre_Image`フォルダにお好きな写真をアップロードしてください。Finder(Mac)、Exploler(Windows)からまとめてアップロードすることができます。  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/Image.GIF">
</div>  

写真のアップロードが終了すると、ターミナルに戻り、以下のように入力してください。  
```
python3 face_recognization.py
```
すると、Post_Imageに顔認識の処理がされたイメージが保存されます。  
同様に  
```
python3 object_detection.py
```
すると、Post_Imageに物体認識の処理がされたイメージが保存されます。  

#### 備忘録  
Intel Dev Cloud for the Edgeでフォルダをまとめて削除したい場合は以下のコマンドを打てば良い。
```
rm -rf フォルダ名
```  

### 実行方法2  
DockerとVisual Studio Codeをしようすることで、ローカル環境でプログラムを実行することができます。ここでは細かな説明を省いているので、エラーが発生した場合などは適宜検索をかけて調べてみてください。  
- [Docker Desktop](https://www.docker.com/products/docker-desktop)  
はじめにDocker Desktopを導入してください。  

- [Visual Studio Code](https://code.visualstudio.com/download)(以下VS Code)  
次にVisual Studio Codeを導入し、ExtensionにDockerを導入してください。(チェック用)  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/Docker_Extension.png">
</div>  

また、Remote Developmentをインストールしてください。これはDockerを用いた開発で非常に使いやすい拡張機能です。  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/Dev_Container.png">
</div>  

Remote Developmentを導入し終わった後、お好みのディレクトリに
```
git clone https://github.com/yusuke-1105/Object_Detection
cd Object_Detection
wget https://www.dropbox.com/s/6dlv9448ssy8lki/Archive.zip?dl=0 -O Models/Archive.zip
unzip Models/Archive.zip -d Models/
```
して、dockerディレクトリ配下の「.devcontainer」を主ディレクトリ配下(このREADMEやプログラムがあるところ)に移動させてください。
dockerディレクトリ配下の「dockerfile」の43行目、
```docker
ARG package_url=<your_package_url>
```  
`<your_package_url>`のところに[Intel® Distribution of OpenVINO™ Toolkit Download Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)で取得したURLを入力してください。「<>」は要りません。  
入力後、VS Codeのコマンドパレット(Ctrl(Command)+shift+P)を開いてください。そこで「Open Folder in Container」をクリックしてください。  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/docker_extension_func.png">
</div>  
適宜フォルダを選択してください。そうすると、Dockerのビルドが始まるので、少し(30~90分)待つとビルドが完成して、プログラムが使えるようになります。  


## モデルについて  
今回使用しているモデルはYoloV3を使用して、OpenVINOで使用できるようにIR形式に変換して使用しています。そのコード等は`Colab`配下の[yolo_to_IR.ipynb](https://github.com/yusuke-1105/Object_Detection/blob/master/Colab/yolo_to_IR.ipynb)に書かれており、Google Claboratoryで使用できるようになっていますので、使ってみてください。  
またyoloにはtiny版があり、それはモデルの容量が小さいため推論が高速に実行できる仕様となっています。(ただし、精度は落ちます。)  
もし、そちらを使用してみたい場合は上記のファイル中にその方法を示してありますので、そちらをご覧ください。

## 参照文献  
[YOLO and Tiny-YOLO object detection on the Raspberry Pi and Movidius NCS](https://www.pyimagesearch.com/2020/01/27/yolo-and-tiny-yolo-object-detection-on-the-raspberry-pi-and-movidius-ncs/)  
[open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo)  
[Converting a TensorFlow* Model](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)  
[YOLOv3 を OpenVINO™ ツールキットで使用する IR 形式へ変換してみよう](https://macnicago.zendesk.com/hc/ja/articles/360042709871-YOLOv3-を-OpenVINO-ツールキットで使用する-IR-形式へ変換してみよう)  
[Install Intel® Distribution of OpenVINO™ toolkit for Linux* from a Docker* Image](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html)  
[Introduction to Intel® Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Introduction.html)
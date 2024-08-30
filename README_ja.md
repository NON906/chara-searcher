# chara-searcher

このブランチは[DuckDuckGo](https://duckduckgo.com/)の画像検索に対応したバージョンです。  
以下はこのブランチでの説明になります。

## インストール・実行方法

### 単独で使用する

condaコマンド（Anaconda）を使用できる場合は、以下のコマンドでインストールできます。

```
git clone https://github.com/NON906/chara-searcher.git
cd chara-searcher
git checkout duckduckgo
conda create -n chara-searcher python=3.10
conda activate chara-searcher
conda install "pytorch<=2.1" torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python install.py
```

インストール後は以下のコマンドで実行できます。

```
conda activate chara-searcher
python standalone_ui.py
```

condaを使用しない場合は、Python本体とpytorch（とcuda）をインストールした後に、``python install.py``を実行してください。  
なお、pytorchのインストール方法は[こちら](https://pytorch.org/get-started/locally/)を確認してください。  
インストール後は``python standalone_ui.py``で実行できます。

### Stable Diffusion web UIにインストールする

1. webuiを起動し、「拡張機能(Extensions)」の「URLからインストール(Install from URL)」から以下の内容を入力し、インストールしてください。  
拡張機能のリポジトリのURL: ``https://github.com/NON906/chara-searcher.git``  
Specific branch name: ``duckduckgo``

2. 「拡張機能(Extensions)」の「インストール済(Installed)」の「適用してUIを再起動(Apply and restart UI)」をクリックし、再起動してください。

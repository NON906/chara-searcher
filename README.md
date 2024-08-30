# chara-searcher

This branch is the version that supports image search on [DuckDuckGo](https://duckduckgo.com/).  
Below is the explanation for this branch.

## How to Install and Run

### Use standalone

If you can use the conda command (Anaconda), you can install it with the command below.

```
git clone https://github.com/NON906/chara-searcher.git
cd chara-searcher
git checkout duckduckgo
conda create -n chara-searcher python=3.10
conda activate chara-searcher
conda install "pytorch<=2.1" torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python install.py
```

After installation, you can run it with the following command.

```
conda activate chara-searcher
python standalone_ui.py
```

If you don't use conda, run ``python install.py`` after installing Python  and pytorch (and cuda).   
How to install pytorch is [HERE](https://pytorch.org/get-started/locally/).   
After installation, you can run it with ``python standalone_ui.py``.

### Install on Stable Diffusion web UI

1. Start webui and enter the following from "Install from URL" in "Extensions" to install.  
URL for extension's git repository: ``https://github.com/NON906/chara-searcher.git``  
Specific branch name: ``duckduckgo``

2. Click "Apply and restart UI" under "Extensions" -> "Installed", and restart.

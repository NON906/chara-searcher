# -*- coding: utf-8 -*-

import os
import shutil

try:
    import launch
    platform = 'sd-webui'
except:
    platform = 'standalone'

default_current_dir = os.getcwd()

if platform == 'sd-webui':
    os.chdir(os.path.dirname(__file__))
    if not os.path.isdir('CartoonSegmentation/.git'):
        shutil.rmtree('CartoonSegmentation')
        launch.run('git clone https://github.com/CartoonSegmentation/CartoonSegmentation.git')
    if not os.path.isdir('wd14-tagger-standalone/.git'):
        shutil.rmtree('wd14-tagger-standalone')
        launch.run('git clone https://github.com/corkborg/wd14-tagger-standalone.git')

    if not launch.is_installed('openmim'):
        launch.run_pip('install -U openmim', 'openmim')
    launch.run(f'"{launch.python}" -m mim install mmengine')
    launch.run(f'"{launch.python}" -m mim install "mmcv>=2.0.0"')
    launch.run(f'"{launch.python}" -m mim install mmdet')

    if not launch.is_installed('onnxruntime') and not launch.is_installed('onnxruntime-gpu'):
        pip_list_str = launch.run(f'"{launch.python}" -m pip list')
        pip_list_lines = pip_list_str.splitlines()
        torch_lines = [item for item in pip_list_lines if item.startswith('torch')]
        torch_version = None
        if torch_lines and len(torch_lines) > 0:
            torch_version = torch_lines[0].split()[-1]
        if torch_version is not None and '+cu' in torch_version:
            cuda_version = torch_version.split('+cu')[-1]
            if cuda_version[:2] == '12':
                launch.run_pip('install onnxruntime-gpu==1.17.1 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/', 'onnxruntime-gpu')
            else:
                launch.run_pip('install onnxruntime-gpu==1.17.1', 'onnxruntime-gpu')
        else:
            launch.run_pip('install onnxruntime==1.17.3', 'onnxruntime')

    if not launch.is_installed('chara-searcher requirements'):
        launch.run_pip('install -r requirements.txt', 'chara-searcher requirements')

os.chdir(os.path.join(os.path.dirname(__file__), 'CartoonSegmentation'))
if not os.path.isfile('models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'):
    os.makedirs('models', exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="dreMaz/AnimeInstanceSegmentation", local_dir="models/AnimeInstanceSegmentation")

os.chdir(default_current_dir)
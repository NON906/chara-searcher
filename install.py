# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import sys

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
    launch.run(f'"{launch.python}" -m mim install mmengine "mmcv>=2.0.0" mmdet')

    pip_list_str = launch.run(f'"{launch.python}" -m pip list')
    pip_list_lines = pip_list_str.splitlines()

    if not launch.is_installed('onnxruntime') and not launch.is_installed('onnxruntime-gpu'):
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
        transformers_lines = [item for item in pip_list_lines if item.startswith('transformers')]
        transformers_version = None
        if transformers_lines and len(transformers_lines) > 0:
            transformers_version = transformers_lines[0].split()[-1]
        if transformers_version is None:
            launch.run_pip('install -r requirements.txt', 'chara-searcher requirements')
        else:
            launch.run_pip('install transformers==' + transformers_version + ' -r requirements.txt', 'chara-searcher requirements')

elif platform == 'standalone':
    os.chdir(os.path.dirname(__file__))
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'])

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'openmim'])
    subprocess.run([sys.executable, '-m', 'mim', 'install', 'mmengine'])
    subprocess.run([sys.executable, '-m', 'mim', 'install', 'mmcv>=2.0.0'])
    subprocess.run([sys.executable, '-m', 'mim', 'install', 'mmdet'])

    conda_list_str = subprocess.run(['conda', 'list'], capture_output=True, text=True, shell=True).stdout
    conda_list_lines = conda_list_str.splitlines()
    cuda_lines = [item for item in conda_list_lines if item.startswith('pytorch-cuda')]
    if cuda_lines and len(cuda_lines) > 0:
        cuda_version = cuda_lines[0].split()[1]
        if cuda_version[:3] == '12.':
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu==1.17.1', '--extra-index-url', 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/'])
        else:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu==1.17.1'])
    else:
        pip_list_str = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True).stdout
        pip_list_lines = pip_list_str.splitlines()
        torch_lines = [item for item in pip_list_lines if item.startswith('torch')]
        torch_version = None
        if torch_lines and len(torch_lines) > 0:
            torch_version = torch_lines[0].split()[-1]
        if torch_version is not None and '+cu' in torch_version:
            cuda_version = torch_version.split('+cu')[-1]
            if cuda_version[:2] == '12':
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu==1.17.1', '--extra-index-url', 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/'])
            else:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu==1.17.1'])
        else:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime==1.17.3'])

    subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers>=4.34.0'])

    subprocess.run([sys.executable, '-m', 'pip', 'install', 'gradio==3.41.2'])

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

os.chdir(os.path.join(os.path.dirname(__file__), 'CartoonSegmentation'))
if not os.path.isfile('models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'):
    os.makedirs('models', exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="dreMaz/AnimeInstanceSegmentation", local_dir="models/AnimeInstanceSegmentation")

os.chdir(default_current_dir)
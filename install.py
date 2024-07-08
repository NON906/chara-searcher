# -*- coding: utf-8 -*-

import os

if __name__ != '__main__':
    import launch
    default_current_dir = os.getcwd()

    os.chdir(os.path.dirname(__file__))
    launch.run('git submodule update --init --recursive')
    
    os.chdir(os.path.join(os.path.dirname(__file__), 'CartoonSegmentation'))
    if not launch.is_installed('openmim'):
        launch.run_pip('install -U openmim', 'openmim')
    launch.run(f'"{launch.python}" -m mim install mmengine')
    launch.run(f'"{launch.python}" -m mim install "mmcv>=2.0.0"')
    launch.run(f'"{launch.python}" -m mim install mmdet')

    os.chdir(os.path.dirname(__file__))
    if not launch.is_installed('chara-searcher requirements'):
        launch.run_pip('install -r requirements.txt', 'chara-searcher requirements')

    os.chdir(os.path.join(os.path.dirname(__file__), 'CartoonSegmentation'))
    os.makedirs('models', exist_ok=True)
    launch.run('huggingface-cli lfs-enable-largefiles . && git clone https://huggingface.co/dreMaz/AnimeInstanceSegmentation models/AnimeInstanceSegmentation')

    os.chdir(default_current_dir)
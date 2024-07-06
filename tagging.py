# -*- coding: utf_8 -*-

import subprocess

def tagging_main(dir='src_images', threshold=0.35):
    command = ['python', 'wd14-tagger-standalone/run.py', '--dir', dir, '--model', 'wd-v1-4-moat-tagger.v2', '--threshold', str(threshold)]
    subprocess.run(command)

if __name__ == '__main__':
    tagging_main()
# -*- coding: utf_8 -*-

import os
import argparse
import subprocess

def frame_extraction_main(targets):
    os.makedirs('org_images', exist_ok=True)
    for loop, target in enumerate(targets):
        command = ['ffmpeg', '-i', target, '-skip_frame', 'nokey', '-r', '0.25', 'org_images/' + str(loop) + '_%09d.png']
        subprocess.run(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='*')
    args = parser.parse_args()

    frame_extraction_main(args.paths)
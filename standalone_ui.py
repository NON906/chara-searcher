# -*- coding: utf-8 -*-

import gradio as gr
import argparse
import os
import shutil
import av
from tqdm import tqdm
import cv2

from segmentation import segmentation_main, segmentation_single
from tagging import tagging_main
from calc_embedding import calc_embedding_main

def get_unique_dir(data_name):
    src_images_dir_base = os.path.join('outputs', 'image_search_datas', data_name)
    src_images_dir = src_images_dir_base
    dir_loop = 0
    while os.path.isdir(src_images_dir):
        dir_loop += 1
        src_images_dir = src_images_dir_base + '_' + str(dir_loop)
    return src_images_dir

def upload_dir_files(files, data_name):
    if data_name is None or data_name == '':
        data_name = 'Untitled'
    src_images_dir = get_unique_dir(data_name)

    file_paths = [file.name for file in files]
    
    os.makedirs(src_images_dir, exist_ok=True)

    print('Processing segmentation.')
    segmentation_main(file_paths, src_images_dir)

    print('Processing tagging.')
    tagging_main(src_images_dir)

    print('Processing calc embedding.')
    calc_embedding_main(src_images_dir)

    return ''

def upload_video_file(file, data_name, span=4.0, fps=-1.0):
    if type(file) == list:
        file = file[0]
    if data_name is None or data_name == '':
        data_name = os.path.splitext(os.path.basename(file.name))[0]
    src_images_dir = get_unique_dir(data_name)

    os.makedirs(src_images_dir, exist_ok=True)

    if fps > 0.0:
        span = 1.0 / str(fps)

    print('Processing segmentation.')
    container = av.open(file.name, options={'skip_frame': 'nokey'})
    stream = container.streams.video[0]
    next_time = 0.0
    frame_count = 0
    for frame in tqdm(container.decode(video=0), total=stream.frames):
        if type(frame) == av.video.frame.VideoFrame and next_time <= frame.time:
            frame_np = frame.to_ndarray(format='bgr24')
            trim_dsts = segmentation_single(frame_np)
            for ii, trim_dst in enumerate(trim_dsts):
                if trim_dst.shape[0] > 0 and trim_dst.shape[1] > 0:
                    save_filename = os.path.join(src_images_dir, str(frame_count) + '_' + str(ii) + '.png')
                    cv2.imwrite(save_filename, trim_dst)
            next_time += span
            frame_count += 1

    print('Processing tagging.')
    tagging_main(src_images_dir)

    print('Processing calc embedding.')
    calc_embedding_main(src_images_dir)

    return ''

def main_ui():
    with gr.Blocks() as block_interface:
        with gr.Row():
            gr.Markdown(value='## Load Datas')
        with gr.Row():
            with gr.Column():
                upload_data_name = gr.Textbox(label='Data Name')
                upload_dir_btn = gr.UploadButton(label='Upload Images Directory', file_count='directory')
                upload_video_file_btn = gr.UploadButton(label='Upload Video Files')
            with gr.Column():
                target_datas = gr.Dropdown(choices=[], value=[], label='Target Datas', multiselect=True)

        upload_dir_btn.upload(fn=upload_dir_files, inputs=[upload_dir_btn, upload_data_name], outputs=upload_data_name)
        upload_video_file_btn.upload(fn=upload_video_file, inputs=[upload_video_file_btn, upload_data_name], outputs=upload_data_name)

    return block_interface

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_browser_open', action='store_true')
    args = parser.parse_args()

    block_interface = main_ui()

    block_interface.launch(inbrowser=(not args.disable_browser_open))
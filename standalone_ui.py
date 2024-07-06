# -*- coding: utf-8 -*-

import gradio as gr
import argparse
import os
import shutil

from segmentation import segmentation_main
from tagging import tagging_main
from calc_embedding import calc_embedding_main

def main_ui():
    with gr.Blocks() as block_interface:
        with gr.Row():
            gr.Markdown(value='## Load Datas')
        with gr.Row():
            with gr.Column():
                upload_data_name = gr.Textbox(label='Data Name')
                upload_dir_btn = gr.UploadButton(label='Upload Images Directory', file_count='directory')
                upload_files_btn = gr.UploadButton(label='Upload Video Files')
            with gr.Column():
                target_datas = gr.Dropdown(choices=[], value=[], label='Target Datas', multiselect=True)

        def upload_dir_files(files, data_name):
            if data_name is None or data_name == '':
                data_name = 'Untitled'
            src_images_dir_base = os.path.join('outputs', 'image_search_datas', data_name)
            src_images_dir = src_images_dir_base
            dir_loop = 0
            while os.path.isdir(src_images_dir):
                dir_loop += 1
                src_images_dir = src_images_dir_base + '_' + str(dir_loop)

            file_paths = [file.name for file in files]
            
            os.makedirs(src_images_dir, exist_ok=True)

            print('Processing segmentation.')
            segmentation_main(file_paths, src_images_dir)

            print('Processing tagging.')
            tagging_main(src_images_dir)

            print('Processing calc embedding.')
            calc_embedding_main(src_images_dir)

            return ''

        upload_dir_btn.upload(fn=upload_dir_files, inputs=[upload_dir_btn, upload_data_name], outputs=upload_data_name)

    return block_interface

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_browser_open', action='store_true')
    args = parser.parse_args()

    block_interface = main_ui()

    block_interface.launch(inbrowser=(not args.disable_browser_open))
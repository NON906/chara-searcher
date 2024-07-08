# -*- coding: utf-8 -*-

import gradio as gr
import argparse
import os
import shutil
import av
from tqdm import tqdm
import cv2
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import json
import datetime

from segmentation import segmentation_main, segmentation_single
from tagging import tagging_main
from calc_embedding import calc_embedding_main

img_model = None
loaded_target_datas = None
embedding_file_datas = None
sorted_similarities_index = None
sorted_similarities = None

def get_unique_dir(data_name):
    src_images_dir_base = os.path.join('outputs', 'image_search_datas', data_name)
    src_images_dir = src_images_dir_base
    dir_loop = 0
    while os.path.isdir(src_images_dir):
        dir_loop += 1
        src_images_dir = src_images_dir_base + '_' + str(dir_loop)
    return src_images_dir

def get_target_datas_choices():
    dir_path = os.path.join('outputs', 'image_search_datas')
    target_datas_choices = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f, 'embedding.json'))]
    return target_datas_choices

def upload_dir_files(files, data_name, target_datas):
    if data_name is None or data_name == '':
        data_name = 'Untitled'
    src_images_dir = get_unique_dir(data_name)

    file_paths = [file.name for file in files]
    
    os.makedirs(src_images_dir, exist_ok=True)

    print('* Processing segmentation.')
    segmentation_main(file_paths, src_images_dir)

    print('* Processing tagging.')
    tagging_main(src_images_dir)

    print('* Processing calc embedding.')
    calc_embedding_main(src_images_dir)

    return '', gr.update(choices=get_target_datas_choices(), value=target_datas + [os.path.basename(src_images_dir), ])

def upload_video_file(file, data_name, target_datas, span=4.0, fps=-1.0):
    if type(file) == list:
        file = file[0]
    if data_name is None or data_name == '':
        data_name = os.path.splitext(os.path.basename(file.name))[0]
    src_images_dir = get_unique_dir(data_name)

    os.makedirs(src_images_dir, exist_ok=True)

    if fps > 0.0:
        span = 1.0 / str(fps)

    print('* Processing segmentation.')
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

    print('* Processing tagging.')
    tagging_main(src_images_dir)

    print('* Processing calc embedding.')
    calc_embedding_main(src_images_dir)

    return '', gr.update(choices=get_target_datas_choices(), value=target_datas + [os.path.basename(src_images_dir), ])

def load_target_datas(target_datas):
    global img_model, embedding, embedding_file_datas, loaded_target_datas
    if loaded_target_datas == target_datas:
        return

    if img_model is None:
        img_model = SentenceTransformer('clip-ViT-B-32')

    embedding_array = []
    embedding_file_datas = []

    for target_data in target_datas:
        print('* Load "' + target_data + '".')
        dir_base = os.path.join('outputs', 'image_search_datas', target_data)
        embedding_array.append(np.load(os.path.join(dir_base, 'embedding.npz'))['embedding'])
        with open(os.path.join(dir_base, 'embedding.json'), 'r') as f:
            add_embedding_files = json.load(f)['files']
        for file in tqdm(add_embedding_files):
            txt_path = os.path.join(dir_base, os.path.splitext(file)[0] + '.txt')
            with open(txt_path, 'r') as f:
                embedding_file_datas.append((os.path.join(dir_base, file), f.read()))
    embedding = np.concatenate(embedding_array, 0)
    loaded_target_datas = target_datas

def search_filter(threshold, positive_keywords, negative_keywords, export_exclude_tags, target_datas=None):
    global embedding_file_datas, sorted_similarities_index, sorted_similarities
    if target_datas is not None:
        load_target_datas(target_datas)

    if sorted_similarities_index is None or sorted_similarities is None:
        sorted_similarities_index = np.arange(len(embedding_file_datas) - 1, -1, -1)
        sorted_similarities = np.ones_like(sorted_similarities_index)

    positive_keywords = positive_keywords.split(',')
    positive_keywords = [k.strip() for k in positive_keywords]
    positive_keywords = [k for k in positive_keywords if k != '']
    negative_keywords = negative_keywords.split(',')
    negative_keywords = [k.strip() for k in negative_keywords]
    negative_keywords = [k for k in negative_keywords if k != '']
    ret = []
    tags = {}
    max_tags_count = 0
    for loop in range(sorted_similarities_index.shape[0] - 1, -1, -1):
        index = sorted_similarities_index[loop]
        similarity = sorted_similarities[loop]
        if similarity < threshold / 100.0:
            break
        do_append = True
        for keyword in positive_keywords:
            if not keyword in embedding_file_datas[index][1]:
                do_append = False
                break
        for keyword in negative_keywords:
            if keyword in embedding_file_datas[index][1]:
                do_append = False
                break
        if do_append:
            ret.append(embedding_file_datas[index][0])
            file_tags = embedding_file_datas[index][1].split(',')
            file_tags = [k.strip() for k in file_tags]
            file_tags = [k for k in file_tags if k != '']
            for tag in file_tags:
                if tag in tags:
                    tags[tag] += 1
                else:
                    tags[tag] = 1
                if tags[tag] > max_tags_count:
                    max_tags_count = tags[tag]

    tags_count_list = [[] for _ in range(max_tags_count + 1)]
    for tag, count in tags.items():
        tags_count_list[count].append(tag + ' (' + str(count) + ')')
    tags_list = []
    tags_count_list = tags_count_list[::-1]
    for tags_count_items in tags_count_list:
        tags_list += tags_count_items

    tags_value = []
    for tag_raw in export_exclude_tags:
        tag = ' ('.join(tag_raw.split(' (')[:-1])
        tag_added = False
        for tag_full in tags_list:
            if tag in tag_full:
                tags_value.append(tag_full)
                tag_added = True
                break
        if not tag_added:
            tags_value.append(tag + ' (0)')

    return ret, gr.update(choices=tags_list, value=tags_value)

def search_image_sort(target_datas, image, threshold, positive_keywords, negative_keywords, export_exclude_tags):
    global img_model, embedding, sorted_similarities_index, sorted_similarities
    load_target_datas(target_datas)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    seg_images = segmentation_single(image_bgr)
    if len(seg_images) > 0:
        seg_image = cv2.cvtColor(seg_images[0], cv2.COLOR_BGR2RGB)
    else:
        seg_image = image

    pil_image = Image.fromarray(seg_image)
    image_embedding = img_model.encode(pil_image)

    similarities = img_model.similarity(embedding, image_embedding)
    similarities = np.squeeze(similarities)

    sorted_similarities_index = np.argsort(similarities)
    sorted_similarities = np.sort(similarities)

    result_images, tags_list = search_filter(threshold, positive_keywords, negative_keywords, export_exclude_tags)

    return result_images, seg_image, tags_list

def search_clear_image(threshold, positive_keywords, negative_keywords, target_datas, export_exclude_tags):
    global sorted_similarities_index, sorted_similarities
    sorted_similarities_index = None
    sorted_similarities = None
    result_images, tags_list = search_filter(threshold, positive_keywords, negative_keywords, export_exclude_tags, target_datas)
    return result_images, tags_list

def export(images, dir_name, add_tags, exclude_tags):
    global sorted_similarities_index, embedding_file_datas
    os.makedirs(dir_name, exist_ok=True)

    add_tags = add_tags.split(',')
    add_tags = [k.strip() for k in add_tags]
    add_tags = [k for k in add_tags if k != '']

    exclude_tags = [' ('.join(k.split(' (')[:-1]) for k in exclude_tags]

    for loop in range(len(images)):
        index = sorted_similarities_index[-loop - 1]
        file_tags = embedding_file_datas[index][1].split(',')
        file_tags = [k.strip() for k in file_tags]
        file_tags = [k for k in file_tags if k != '']
        for tag in exclude_tags:
            if tag in file_tags:
                file_tags.remove(tag)
        for tag in add_tags:
            if tag in file_tags:
                file_tags.remove(tag)
        file_tags = add_tags + file_tags

        target_file_base = os.path.join(dir_name, os.path.splitext(os.path.basename(embedding_file_datas[index][0]))[0])
        ext = os.path.splitext(os.path.basename(embedding_file_datas[index][0]))[1]
        target_file = target_file_base + ext
        dir_loop = 0
        while os.path.isfile(target_file):
            dir_loop += 1
            target_file = target_file_base + '_' + str(dir_loop) + ext

        shutil.copy2(embedding_file_datas[index][0], target_file)
        txt_path = os.path.splitext(target_file)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write(', '.join(file_tags))
        
    print('* Finish export.')

def main_ui():
    target_datas_choices = get_target_datas_choices()
    dt_now = datetime.datetime.now()
    save_dt = dt_now.strftime('%Y%m%d_%H%M%S')

    with gr.Blocks() as block_interface:
        with gr.Row():
            gr.Markdown(value='## Load Datas')
        with gr.Row():
            with gr.Column():
                upload_data_name = gr.Textbox(label='Data Name')
                upload_dir_btn = gr.UploadButton(label='Upload Images Directory', file_count='directory')
                upload_video_file_btn = gr.UploadButton(label='Upload Video Files')
            with gr.Column():
                target_datas = gr.Dropdown(choices=target_datas_choices, value=target_datas_choices, label='Target Datas', multiselect=True, interactive=True)
        with gr.Row():
            gr.Markdown(value='## Search And Preview')
        with gr.Row():
            with gr.Column():
                search_image = gr.Image(label='Search Image')
                search_threshold_slider = gr.Slider(label='Search Image Threshold', value=90.0)
            with gr.Column():
                search_positive_keywords = gr.Textbox(label='Search Tags')
                search_negative_keywords = gr.Textbox(label='Negative Tags')
        with gr.Row():
            search_result_gallery = gr.Gallery(label='Search Result', columns=10)
        with gr.Row():
            gr.Markdown(value='## Export')
        with gr.Row():
            with gr.Column():
                export_dir_name = gr.Textbox(label='Export Directory', value='outputs/export_train_datas/' + save_dt, interactive=True)
                export_add_tags = gr.Textbox(label='Additional Tags', value='white background, simple background', interactive=True)
                export_exclude_tags = gr.Dropdown(choices=[], value=[], label='Exclude Tags', multiselect=True, interactive=True)
                export_button = gr.Button(value='Export')

        upload_dir_btn.upload(fn=upload_dir_files, inputs=[upload_dir_btn, upload_data_name, target_datas], outputs=[upload_data_name, target_datas])
        upload_video_file_btn.upload(fn=upload_video_file, inputs=[upload_video_file_btn, upload_data_name, target_datas], outputs=[upload_data_name, target_datas])

        search_image.upload(fn=search_image_sort,
            inputs=[target_datas, search_image, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags],
            outputs=[search_result_gallery, search_image, export_exclude_tags])
        search_image.clear(fn=search_clear_image,
            inputs=[search_threshold_slider, search_positive_keywords, search_negative_keywords, target_datas, export_exclude_tags],
            outputs=[search_result_gallery, export_exclude_tags])
        search_threshold_slider.change(fn=search_filter,
            inputs=[search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags])
        search_positive_keywords.change(fn=search_filter,
            inputs=[search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags])
        search_negative_keywords.change(fn=search_filter,
            inputs=[search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags])

        export_button.click(fn=export, inputs=[search_result_gallery, export_dir_name, export_add_tags, export_exclude_tags])

    return block_interface

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_browser_open', action='store_true')
    args = parser.parse_args()

    block_interface = main_ui()

    block_interface.launch(inbrowser=(not args.disable_browser_open))
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
from sklearn.metrics.pairwise import cosine_similarity
import sys
import logging
from duckduckgo_search import DDGS
import requests
import threading
from urllib.parse import urlparse
import time
import tempfile
import uuid

from segmentation import segmentation_main, segmentation_single, segmentation_unload_net
from tagging import tagging_main, tagging_in_memory, tagging_unload
from calc_embedding import calc_embedding_main, convert_rgba_to_rgb, calc_embedding_in_memory

img_model = None
loaded_target_datas = None
embedding_file_datas = None
sorted_similarities_index = None
sorted_similarities = None
exclude_datas_indexs = []
is_search_state = 'wait'
click_gallery_image_index = 0
embedding = None
#global_platform = 'standalone'
ddg_datas = {}
ddg_target_urls = []
ddg_preview_datas = []
ddg_loading_domains = []
ddg_loading_images = []
temp_dir = tempfile.TemporaryDirectory()

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
    if not os.path.isdir(dir_path):
        return []
    target_datas_choices = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f, 'embedding.json'))]
    return target_datas_choices

def upload_dir_files(files, data_name, target_datas):
    global img_model

    if data_name is None or data_name == '':
        data_name = 'Untitled'
    src_images_dir = get_unique_dir(data_name)

    file_paths = [file.name for file in files]
    
    os.makedirs(src_images_dir, exist_ok=True)

    print('* (step 1/3) Processing segmentation.')
    segmentation_main(file_paths, src_images_dir)

    print('* (step 2/3) Processing tagging.')
    tagging_main(src_images_dir)

    print('* (step 3/3) Processing calc embedding.')
    if img_model is None:
        img_model = SentenceTransformer('clip-ViT-B-32')
    calc_embedding_main(src_images_dir, img_model)

    return '', gr.update(choices=get_target_datas_choices(), value=target_datas + [os.path.basename(src_images_dir), ])

def upload_video_file(file, data_name, target_datas, span=4.0, fps=-1.0):
    global img_model

    if type(file) == list:
        file = file[0]
    if data_name is None or data_name == '':
        data_name = os.path.splitext(os.path.basename(file.name))[0]
    src_images_dir = get_unique_dir(data_name)

    os.makedirs(src_images_dir, exist_ok=True)

    if fps > 0.0:
        span = 1.0 / str(fps)

    print('* (step 1/3) Processing segmentation.')
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

    print('* (step 2/3) Processing tagging.')
    tagging_main(src_images_dir)

    print('* (step 3/3) Processing calc embedding.')
    if img_model is None:
        img_model = SentenceTransformer('clip-ViT-B-32')
    calc_embedding_main(src_images_dir, img_model)

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
        for file_item in tqdm(add_embedding_files):
            file = file_item['path']
            txt_path = os.path.join(dir_base, os.path.splitext(file)[0] + '.txt')
            with open(txt_path, 'r') as f:
                embedding_file_datas.append((os.path.join(dir_base, file), f.read(), file_item['width'], file_item['height']))
    if len(embedding_array) > 0:
        embedding = np.concatenate(embedding_array, 0)
    else:
        embedding = None
    loaded_target_datas = target_datas

def search_filter_main(threshold, positive_keywords, negative_keywords, export_exclude_tags, min_size=128):
    global embedding_file_datas, sorted_similarities_index, sorted_similarities, exclude_datas_indexs

    if sorted_similarities_index is None:
        sorted_similarities_index = np.arange(len(embedding_file_datas) - 1, -1, -1)

    positive_keywords, negative_keywords = keyword_parse(positive_keywords, negative_keywords)
    ret = []
    tags = {}
    max_tags_count = 0
    for loop in range(sorted_similarities_index.shape[0] - 1, -1, -1):
        index = sorted_similarities_index[loop]
        if sorted_similarities is not None:
            similarity = sorted_similarities[loop]
            if similarity < threshold / 100.0:
                break
        if is_targeted_image_judge(index, positive_keywords, negative_keywords, min_size):
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
            if len(ret) % 100 == 0:
                yield ret, gr.update()

    tags_count_list = [[] for _ in range(max_tags_count + 1)]
    for tag, count in tags.items():
        tags_count_list[count].append(tag + ' (' + str(count) + ')')
    tags_list = []
    tags_count_list = tags_count_list[::-1]
    for tags_count_items in tags_count_list:
        tags_list += tags_count_items

    tags_value = []
    if export_exclude_tags is not None:
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

    yield ret, gr.update(choices=tags_list, value=tags_value)

def search_filter(mode, threshold, positive_keywords, negative_keywords, export_exclude_tags, target_datas=None, min_size=128):
    global is_search_state

    if mode != 'Local':
        yield gr.update(), gr.update()
        return

    if target_datas is not None:
        load_target_datas(target_datas)
    
    for ret_files, ret_tags in search_filter_main(threshold, positive_keywords, negative_keywords, export_exclude_tags, min_size):
        yield ret_files, ret_tags
        if is_search_state == 'cancel':
            break

    is_search_state = 'wait'

def search_ddg(mode, ddg_keywords, region='wt-wt', safesearch='off'):
    global ddg_datas, ddg_target_urls, img_model, ddg_loading_domains, ddg_loading_images

    if mode != 'DuckDuckGo':
        return

    ddg_results = DDGS().images(
        keywords=ddg_keywords,
        region=region,
        safesearch=safesearch,
    )

    if img_model is None:
        img_model = SentenceTransformer('clip-ViT-B-32')

    ddg_target_urls = []

    for ddg_result in ddg_results:
        url = ddg_result['thumbnail']
        if not url in ddg_datas or ddg_datas[url] == 'Error':
            ddg_datas[url] = 'Waiting'

    def loading_process(ddg_result):
        global ddg_loading_domains, ddg_loading_images

        url = ddg_result['thumbnail']

        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        while domain in ddg_loading_domains or len(ddg_loading_domains) >= 4:
            time.sleep(0.01)
        ddg_loading_domains.append(domain)

        response = requests.get(url)
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
        if image is None:
            print('error: ' + url)
            ddg_datas[url] = 'Error'
            time.sleep(0.5)
            ddg_loading_domains.remove(domain)
            return False

        ddg_loading_images.append({
            'image': image,
            'url': url,
            'raw_url': ddg_result['image'],
        })

        ddg_loading_domains.remove(domain)
        return True

    is_first_loop = True
    for ddg_result in ddg_results:
        url = ddg_result['thumbnail']
        ddg_target_urls.append(url)
        if ddg_datas[url] == 'Waiting':
            ddg_datas[url] = 'Loading'
            if is_first_loop:
                is_first_loop = not loading_process(ddg_result)
            else:
                thread = threading.Thread(target=loading_process, args=(ddg_result, ))
                thread.start()

def loading_ddg():
    global ddg_loading_images, is_search_state
    
    target_images = ddg_loading_images
    ddg_loading_images = []

    seg_images = []
    urls = []
    raw_urls = []
    for target_data in target_images:
        image = target_data['image']
        url = target_data['url']

        raw_seg_images = segmentation_single(image)
        if is_search_state != 'running':
            return
        for seg_image in raw_seg_images:
            seg_images.append(cv2.cvtColor(seg_image, cv2.COLOR_BGRA2RGBA))
            urls.append(url)
            raw_urls.append(target_data['raw_url'])
        del image

    embeddings, sizes = calc_embedding_in_memory(seg_images, img_model)
    if is_search_state != 'running':
        return

    tags = tagging_in_memory(seg_images)
    if is_search_state != 'running':
        return
    
    data_dict_array = []
    for loop in range(len(seg_images)):
        data_dict = {
            'image': seg_images[loop],
            'embedding': embeddings[loop],
            'size': sizes[loop],
            'tags': tags[loop],
            'raw_url': raw_urls[loop],
            'file_path': None,
        }
        if ddg_datas[urls[loop]] == 'Loading':
            ddg_datas[urls[loop]] = [data_dict, ]
            #print('loaded: ' + urls[loop])
        else:
            ddg_datas[urls[loop]].append(data_dict)

def search_filter_ddg_main(threshold, positive_keywords, negative_keywords, export_exclude_tags, image, min_size=128):
    global ddg_datas, ddg_target_urls, ddg_preview_datas, exclude_datas_indexs

    if image is not None:
        if image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            seg_images = segmentation_single(image_bgr)
            if len(seg_images) > 0:
                seg_image = cv2.cvtColor(seg_images[0], cv2.COLOR_BGRA2RGBA)
            else:
                seg_image = image
        else:
            seg_image = image

        if seg_image.shape[2] == 4:
            search_image = convert_rgba_to_rgb(seg_image)
        else:
            search_image = seg_image

        pil_image = Image.fromarray(search_image)
        image_embedding = img_model.encode(pil_image)
    else:
        seg_image = gr.update()
        image_embedding = None

    positive_keywords, negative_keywords = keyword_parse(positive_keywords, negative_keywords)
    ddg_preview_datas = []
    ret_list = []
    tags = {}
    max_tags_count = 0
    target_urls = ddg_target_urls

    is_finished = False
    while not is_finished:
        is_finished = True

        loading_ddg()

        new_target_urls = []

        for url in target_urls:
            if ddg_datas[url] == 'Waiting' or ddg_datas[url] == 'Loading':
                is_finished = False
                new_target_urls.append(url)
                continue
            elif ddg_datas[url] == 'Error':
                continue
            for data in ddg_datas[url]:
                do_append = True
                for keyword in positive_keywords:
                    if not keyword in data['tags']:
                        do_append = False
                        break
                for keyword in negative_keywords:
                    if keyword in data['tags']:
                        do_append = False
                        break
                if data['size'][0] < min_size or data['size'][1] < min_size:
                    do_append = False
                if not do_append:
                    continue

                if image_embedding is not None:
                    if hasattr(img_model, 'similarity'):
                        similarities = img_model.similarity(data['embedding'], image_embedding)
                    else:
                        similarities = cosine_similarity(np.array([data['embedding'], ]), np.array([image_embedding, ]))
                    similarities = np.squeeze(similarities)
                    if similarities < threshold / 100.0:
                        continue

                ddg_preview_datas.append(data)
                if data['file_path'] is None:
                    ret_image = cv2.cvtColor(data['image'], cv2.COLOR_RGBA2BGRA)
                    ret_file_name = temp_dir.name + "/" + str(uuid.uuid4()) + ".png"
                    cv2.imwrite(ret_file_name, ret_image)
                    data['file_path'] = ret_file_name
                ret_list.append(data['file_path'])

                file_tags = data['tags'].split(',')
                file_tags = [k.strip() for k in file_tags]
                file_tags = [k for k in file_tags if k != '']
                for tag in file_tags:
                    if tag in tags:
                        tags[tag] += 1
                    else:
                        tags[tag] = 1
                    if tags[tag] > max_tags_count:
                        max_tags_count = tags[tag]

        target_urls = new_target_urls

        tags_count_list = [[] for _ in range(max_tags_count + 1)]
        for tag, count in tags.items():
            tags_count_list[count].append(tag + ' (' + str(count) + ')')
        tags_list = []
        tags_count_list = tags_count_list[::-1]
        for tags_count_items in tags_count_list:
            tags_list += tags_count_items

        tags_value = []
        if export_exclude_tags is not None:
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
            
        ret = []
        for loop, val in enumerate(ret_list):
            if not loop in exclude_datas_indexs:
                ret.append(val)

        yield ret, gr.update(choices=tags_list, value=tags_value), seg_image

def search_filter_ddg(mode, threshold, positive_keywords, negative_keywords, export_exclude_tags, image, min_size=128):
    global is_search_state

    if mode != 'DuckDuckGo':
        yield gr.update(), gr.update(), gr.update()
        return

    for ret_files, ret_tags, ret_image in search_filter_ddg_main(threshold, positive_keywords, negative_keywords, export_exclude_tags, image, min_size):
        yield ret_files, ret_tags, ret_image
        if is_search_state == 'cancel':
            break

    is_search_state = 'wait'

def search_cancel():
    global is_search_state
    if is_search_state == 'running':
        is_search_state = 'cancel'

def search_wait():
    global is_search_state
    while is_search_state != 'wait':
        yield
    is_search_state = 'running'

def search_image_sort(mode, target_datas, image):
    global img_model, embedding, sorted_similarities_index, sorted_similarities

    if mode != 'Local':
        return gr.update()

    load_target_datas(target_datas)
    if embedding is None or img_model is None:
        return None

    if image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        seg_images = segmentation_single(image_bgr)
        if len(seg_images) > 0:
            seg_image = cv2.cvtColor(seg_images[0], cv2.COLOR_BGRA2RGBA)
        else:
            seg_image = image
    else:
        seg_image = image

    if seg_image.shape[2] == 4:
        search_image = convert_rgba_to_rgb(seg_image)
    else:
        search_image = seg_image

    pil_image = Image.fromarray(search_image)
    image_embedding = img_model.encode(pil_image)

    if hasattr(img_model, 'similarity'):
        similarities = img_model.similarity(embedding, image_embedding)
    else:
        similarities = cosine_similarity(embedding, np.array([image_embedding, ]))
    similarities = np.squeeze(similarities)

    sorted_similarities_index = np.argsort(similarities)
    sorted_similarities = np.sort(similarities)

    return seg_image

def search_clear_image():
    global sorted_similarities_index, sorted_similarities
    sorted_similarities_index = None
    sorted_similarities = None

    return None

def is_targeted_image_judge(index, positive_keywords, negative_keywords, min_size):
    global exclude_datas_indexs, embedding_file_datas

    do_append = True
    for keyword in positive_keywords:
        if not keyword in embedding_file_datas[index][1]:
            do_append = False
            break
    for keyword in negative_keywords:
        if keyword in embedding_file_datas[index][1]:
            do_append = False
            break
    if index in exclude_datas_indexs:
        do_append = False
    if embedding_file_datas[index][2] < min_size or embedding_file_datas[index][3] < min_size:
        do_append = False
    return do_append

def keyword_parse(positive_keywords, negative_keywords):
    positive_keywords = positive_keywords.split(',')
    positive_keywords = [k.strip() for k in positive_keywords]
    positive_keywords = [k for k in positive_keywords if k != '']
    negative_keywords = negative_keywords.split(',')
    negative_keywords = [k.strip() for k in negative_keywords]
    negative_keywords = [k for k in negative_keywords if k != '']

    return positive_keywords, negative_keywords

def export(mode, images, dir_name, add_tags, exclude_tags, positive_keywords, negative_keywords, min_size=128, color=[255, 255, 255]):
    global sorted_similarities_index, embedding_file_datas, exclude_datas_indexs

    if mode != 'Local':
        return

    os.makedirs(dir_name, exist_ok=True)

    add_tags = add_tags.split(',')
    add_tags = [k.strip() for k in add_tags]
    add_tags = [k for k in add_tags if k != '']

    exclude_tags = [' ('.join(k.split(' (')[:-1]) for k in exclude_tags]

    positive_keywords, negative_keywords = keyword_parse(positive_keywords, negative_keywords)
    loop = 0
    for _ in images:
        index = sorted_similarities_index[-loop - 1]
        while not is_targeted_image_judge(index, positive_keywords, negative_keywords, min_size):
            loop += 1
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

        if color is None:
            shutil.copy2(embedding_file_datas[index][0], target_file)
        else:
            copy_image = cv2.imread(embedding_file_datas[index][0], -1)
            copy_image = convert_rgba_to_rgb(copy_image, color)
            cv2.imwrite(target_file, copy_image)
        txt_path = os.path.splitext(target_file)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write(', '.join(file_tags))
        
        loop += 1
        
    print('* Finish export.')

def export_ddg(mode, dir_name, add_tags, exclude_tags, color=[255, 255, 255]):
    global ddg_preview_datas, ddg_loading_domains, exclude_datas_indexs

    if mode != 'DuckDuckGo':
        return

    os.makedirs(dir_name, exist_ok=True)

    add_tags = add_tags.split(',')
    add_tags = [k.strip() for k in add_tags]
    add_tags = [k for k in add_tags if k != '']

    exclude_tags = [' ('.join(k.split(' (')[:-1]) for k in exclude_tags]

    file_count = 0
    last_url = ''
    seg_images = []
    image_embeddings = None

    for loop, data in enumerate(ddg_preview_datas):
        if loop in exclude_datas_indexs:
            continue

        image_url = data['raw_url']

        if last_url != image_url:
            parsed_url = urlparse(image_url)
            domain = parsed_url.netloc
            while domain in ddg_loading_domains or len(ddg_loading_domains) >= 4:
                time.sleep(0.01)

            ddg_loading_domains.append(domain)

            response = requests.get(image_url)
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
            if image is None:
                time.sleep(0.5)
                response = requests.get(image_url)
                image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)

            ddg_loading_domains.remove(domain)

            if image is None:
                print('export error: ' + image_url)
                seg_images = [data['image'], ]
                time.sleep(0.5)
            else:
                seg_images = []
                raw_seg_images = segmentation_single(image)
                for seg_image in raw_seg_images:
                    seg_images.append(cv2.cvtColor(seg_image, cv2.COLOR_BGRA2RGBA))
                del image
            
                if len(seg_images) > 1:
                    image_embeddings_list = []
                    for seg_image in seg_images:
                        if seg_image.shape[2] == 4:
                            search_image = convert_rgba_to_rgb(seg_image)
                        else:
                            search_image = seg_image
                        pil_image = Image.fromarray(search_image)
                        image_embeddings_list.append(img_model.encode(pil_image))
                    image_embeddings = np.stack(image_embeddings_list, 0)

            last_url = image_url

        if len(seg_images) > 1:
            if hasattr(img_model, 'similarity'):
                similarities = img_model.similarity(image_embeddings, data['embedding'])
            else:
                similarities = cosine_similarity(image_embeddings, np.array([data['embedding'], ]))
            similarities = np.squeeze(similarities)
            ret_seg_image = seg_images[similarities.argmax()]
        elif len(seg_images) > 0:
            ret_seg_image = seg_images[0]
        if color is not None and ret_seg_image.shape[2] == 4:
            ret_seg_image = convert_rgba_to_rgb(ret_seg_image, color)
        ret_seg_image = cv2.cvtColor(ret_seg_image, cv2.COLOR_RGB2BGR)
                    
        cv2.imwrite(dir_name + "/" + str(file_count) + ".png", ret_seg_image)

        file_tags = data['tags'].split(',')
        file_tags = [k.strip() for k in file_tags]
        file_tags = [k for k in file_tags if k != '']
        for tag in exclude_tags:
            if tag in file_tags:
                file_tags.remove(tag)
        for tag in add_tags:
            if tag in file_tags:
                file_tags.remove(tag)
        txt_path = dir_name + "/" + str(file_count) + '.txt'
        file_tags = add_tags + file_tags
        with open(txt_path, 'w') as f:
            f.write(', '.join(file_tags))

        file_count += 1

    print('* Finish export.')

def pre_click_gallery_image(evt: gr.SelectData):
    global click_gallery_image_index
    click_gallery_image_index = evt.index

def click_gallery_image(mode, func_name, threshold, positive_keywords, negative_keywords, min_size=128):
    global exclude_datas_indexs, sorted_similarities_index, click_gallery_image_index, sorted_similarities
    if func_name == 'Preview':
        return gr.update(), gr.update(), gr.update()

    if mode == 'Local':
        positive_keywords, negative_keywords = keyword_parse(positive_keywords, negative_keywords)
        loop = 0
        for _ in range(click_gallery_image_index + 1):
            index = sorted_similarities_index[-loop - 1]
            while not is_targeted_image_judge(index, positive_keywords, negative_keywords, min_size):
                loop += 1
                index = sorted_similarities_index[-loop - 1]
            loop += 1

        if func_name == 'Exclude':
            exclude_datas_indexs.append(index)
        elif func_name == 'Threshold' and sorted_similarities is not None:
            threshold = float(sorted_similarities[-loop]) * 100.0
        return gr.update(value=[], preview=False), threshold, 'Reset Excluded Images (' + str(len(exclude_datas_indexs)) + ')'
    else:
        exclude_num = 0
        for loop in range(len(ddg_preview_datas)):
            if loop in exclude_datas_indexs:
                exclude_num += 1
                continue
            if func_name == 'Exclude' and loop == click_gallery_image_index + exclude_num:
                exclude_datas_indexs.append(loop)
        return gr.update(value=[], preview=False), threshold, 'Reset Excluded Images (' + str(len(exclude_datas_indexs)) + ')'

def click_reset_exclude_datas():
    global exclude_datas_indexs
    exclude_datas_indexs = []
    return 'Reset Excluded Images (0)'

def unload_models():
    global img_model
    if img_model is not None:
        del img_model
        img_model = None
    segmentation_unload_net()
    tagging_unload()
    print('* Finish unload models.')

def main_ui(platform='standalone'):
    #global global_platform
    #global_platform = platform

    target_datas_choices = get_target_datas_choices()
    dt_now = datetime.datetime.now()
    save_dt = dt_now.strftime('%Y%m%d_%H%M%S')

    with gr.Blocks() as block_interface:
        with gr.Row():
            mode_radio = gr.Radio(label='Mode', choices=['Local', 'DuckDuckGo'], value='Local', interactive=True)
        with gr.Row() as load_datas_title:
            gr.Markdown(value='## Load Datas')
        with gr.Row() as load_datas:
            with gr.Column():
                upload_data_name = gr.Textbox(label='Data Name')
                upload_dir_btn = gr.UploadButton(label='Upload Images Directory', file_count='directory')
                upload_video_file_btn = gr.UploadButton(label='Upload Video File')
            with gr.Column():
                target_datas = gr.Dropdown(choices=target_datas_choices, value=target_datas_choices, label='Target Datas', multiselect=True, interactive=True)
        with gr.Row():
            gr.Markdown(value='## Search')
        with gr.Row():
            with gr.Column():
                search_image = gr.Image(label='Search Image')
                search_threshold_slider = gr.Slider(label='Search Image Threshold', value=90.0)
                search_stop_btn = gr.Button(value='Stop Search')
            with gr.Column():
                ddg_search_keywords = gr.Textbox(label='DuckDuckGo Search Keywords', visible=False)
                search_positive_keywords = gr.Textbox(label='Search Tags')
                search_negative_keywords = gr.Textbox(label='Negative Tags')
                search_gallery_func_radio = gr.Radio(label='Click on the Image', choices=['Preview', 'Exclude', 'Threshold'], value='Preview', interactive=True)
                search_exclude_reset_btn = gr.Button(value='Reset Excluded Images (0)')
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
        with gr.Row():
            gr.Markdown(value='## Others')
        with gr.Row():
            unload_button = gr.Button(value='Unload Models')

        upload_dir_btn.upload(fn=upload_dir_files, inputs=[upload_dir_btn, upload_data_name, target_datas], outputs=[upload_data_name, target_datas])
        upload_video_file_btn.upload(fn=upload_video_file, inputs=[upload_video_file_btn, upload_data_name, target_datas], outputs=[upload_data_name, target_datas])

        search_image.upload(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_image_sort,
            inputs=[mode_radio, target_datas, search_image],
            outputs=search_image
        ).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_image.clear(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_clear_image,
            outputs=search_image
        ).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_threshold_slider.input(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_positive_keywords.submit(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_positive_keywords.blur(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_negative_keywords.submit(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_negative_keywords.blur(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, target_datas],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )

        ddg_search_keywords.submit(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_ddg,
            inputs=[mode_radio, ddg_search_keywords]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        ddg_search_keywords.blur(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=search_ddg,
            inputs=[mode_radio, ddg_search_keywords]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )

        search_result_gallery.select(fn=pre_click_gallery_image, queue=False).then(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=click_gallery_image,
            inputs=[mode_radio, search_gallery_func_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords],
            outputs=[search_result_gallery, search_threshold_slider, search_exclude_reset_btn],
        ).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_exclude_reset_btn.click(fn=search_cancel, queue=False).then(fn=search_wait).then(fn=click_reset_exclude_datas,
            outputs=search_exclude_reset_btn,
        ).then(fn=search_filter,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags],
            outputs=[search_result_gallery, export_exclude_tags]
        ).then(fn=search_filter_ddg,
            inputs=[mode_radio, search_threshold_slider, search_positive_keywords, search_negative_keywords, export_exclude_tags, search_image],
            outputs=[search_result_gallery, export_exclude_tags, search_image]
        )
        search_gallery_func_radio.input(fn=lambda f: gr.update(allow_preview= f == 'Preview', preview=False),
            inputs=search_gallery_func_radio,
            outputs=search_result_gallery,
            queue=False)

        search_stop_btn.click(fn=search_cancel, queue=False)

        export_button.click(fn=search_cancel, queue=False
        ).then(fn=export, inputs=[mode_radio, search_result_gallery, export_dir_name, export_add_tags, export_exclude_tags, search_positive_keywords, search_negative_keywords]
        ).then(fn=export_ddg,
            inputs=[mode_radio, export_dir_name, export_add_tags, export_exclude_tags]
        )

        unload_button.click(fn=unload_models)

        def on_load():
            return target_datas_choices, 'outputs/export_train_datas/' + save_dt
        block_interface.load(fn=on_load, outputs=[target_datas, export_dir_name])

        def on_change_mode_radio(mode):
            is_local = mode == 'Local'
            if is_local:
                search_gallery_update = gr.update(choices=['Preview', 'Exclude', 'Threshold'])
            else:
                search_gallery_update = gr.update(choices=['Preview', 'Exclude'])
            return gr.update(visible=is_local), gr.update(visible=is_local), gr.update(visible=not is_local), search_gallery_update
        mode_radio.input(fn=on_change_mode_radio,
            inputs=mode_radio,
            outputs=[load_datas_title, load_datas, ddg_search_keywords, search_gallery_func_radio])

    return block_interface

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_browser_open', action='store_true')
    args = parser.parse_args()

    block_interface = main_ui()
    block_interface.queue()
    block_interface.launch(inbrowser=(not args.disable_browser_open))
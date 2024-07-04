# -*- coding: utf_8 -*-

import sys

import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

sys.path.append('CartoonSegmentation')
from animeinsseg import AnimeInsSeg, AnimeInstances

def main(mask_thres=0.6, instance_thres=0.1, padding_size=0.1):
    os.makedirs('src_images', exist_ok=True)

    refine_kwargs = {'refine_method': 'refinenet_isnet'}
    ckpt = r'models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'
    os.chdir('CartoonSegmentation')
    net = AnimeInsSeg(ckpt, mask_thr=mask_thres, refine_kwargs=refine_kwargs)
    os.chdir('..')

    filenames = glob.glob('org_images/**/*.*', recursive=True)
    for filename in tqdm(filenames, total=len(filenames)):
        img = cv2.imread(filename, -1)
        src = img[:,:,0:3]
        instances: AnimeInstances = net.infer(
            src,
            output_type='numpy',
            pred_score_thr=instance_thres
        )
        for ii, (xywh, mask) in enumerate(zip(instances.bboxes, instances.masks)):
            p = mask.astype(np.float32)
            if img.shape[2] == 4:
                p = p * img[:,:,3] / 255.0
            p = np.stack([p, p, p], 2)
            dst = np.ones_like(src) * 255.0
            dst *= 1.0 - p
            dst += src * p
            dst = dst.astype(src.dtype)

            left_x = int(xywh[0] - padding_size * xywh[2])
            if left_x < 0:
                left_x = 0
            right_x = int(xywh[2] + xywh[0] + padding_size * xywh[2])
            if right_x > dst.shape[0]:
                right_x = dst.shape[0]
            top_y = int(xywh[1] - padding_size * xywh[3])
            if top_y < 0:
                top_y = 0
            bottom_y = int(xywh[3] + xywh[1] + padding_size * xywh[3])
            if bottom_y > dst.shape[1]:
                bottom_y = dst.shape[1]

            trim_dst = dst[left_x:right_x, top_y:bottom_y]
            if trim_dst.shape[0] > 0 and trim_dst.shape[1] > 0:
                save_filename = 'src_images/' + os.path.splitext(filename)[0].replace('\\', '_').replace('/', '_') + '_' + str(ii) + '.png'
                #print(save_filename)
                cv2.imwrite(save_filename, trim_dst)

if __name__ == '__main__':
    main()
# -*- coding: utf_8 -*-

import sys

import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import torch

sys.path.append('CartoonSegmentation')
from animeinsseg import AnimeInsSeg, AnimeInstances, prepare_refine_batch

class AnimeInsSegSmooth(AnimeInsSeg):
    def _postprocess_refine(self, instances: AnimeInstances, img: np.ndarray, refine_size: int = 720, max_refine_batch: int = 4, **kwargs):
        
        if instances.is_empty:
            return
        
        segs = instances.masks
        is_tensor = instances.is_tensor
        if is_tensor:
            segs = segs.cpu().numpy()
        segs = segs.astype(np.float32)
        im_h, im_w = img.shape[:2]
        
        masks = []
        with torch.no_grad():
            for batch, (pt, pb, pl, pr) in prepare_refine_batch(segs, img, max_refine_batch, self.device, refine_size):
                preds = self.refinenet(batch)[0][0].sigmoid()
                if pb == 0:
                    pb = -im_h
                if pr == 0:
                    pr = -im_w
                preds = preds[..., pt: -pb, pl: -pr]
                preds  = torch.nn.functional.interpolate(preds, (im_h, im_w), mode='bilinear', align_corners=True)
                masks.append(preds.cpu()[:, 0])

        masks = torch.concat(masks, dim=0).to(self.device)
        if not is_tensor:
            masks = masks.cpu().numpy()
        instances.masks = masks

def main(mask_thres=0.6, instance_thres=0.3, padding_size=0.1):
    os.makedirs('src_images', exist_ok=True)

    refine_kwargs = {'refine_method': 'refinenet_isnet'}
    ckpt = r'models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'
    os.chdir('CartoonSegmentation')
    net = AnimeInsSegSmooth(ckpt, refine_kwargs=refine_kwargs)
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
        if instances.bboxes is None or instances.masks is None:
            continue
        for ii, (xywh, mask) in enumerate(zip(instances.bboxes, instances.masks)):
            p = (mask.astype(np.float32) - mask_thres) / (1.0 - mask_thres)
            p = p.clip(0.0, 1.0)
            if img.shape[2] == 4:
                p = p * img[:,:,3] / 255.0
            p = np.stack([p, p, p], 2)
            dst = np.ones_like(src) * 255.0
            dst *= 1.0 - p
            dst += src * p
            dst = dst.astype(src.dtype)

            left_x = int(xywh[0] - padding_size * xywh[2])
            right_x = int(xywh[2] + xywh[0] + padding_size * xywh[2])
            if right_x < left_x:
                 left_x, right_x = right_x, left_x
            if left_x < 0:
                left_x = 0
            if right_x > dst.shape[1]:
                right_x = dst.shape[1]
            top_y = int(xywh[1] - padding_size * xywh[3])
            bottom_y = int(xywh[3] + xywh[1] + padding_size * xywh[3])
            if bottom_y < top_y:
                top_y, bottom_y = bottom_y, top_y
            if top_y < 0:
                top_y = 0
            if bottom_y > dst.shape[0]:
                bottom_y = dst.shape[0]

            trim_dst = dst[top_y:bottom_y, left_x:right_x]

            if trim_dst.shape[0] > 0 and trim_dst.shape[1] > 0:
                save_filename = 'src_images/' + os.path.splitext(filename)[0].replace('\\', '_').replace('/', '_') + '_' + str(ii) + '.png'
                #print(save_filename)
                cv2.imwrite(save_filename, trim_dst)

if __name__ == '__main__':
    main()
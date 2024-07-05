# -*- coding: utf_8 -*-

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import os
import shutil

def calc_embedding(filenames, img_model):
    embedding_array = []
    for filename in tqdm(filenames, total=len(filenames)):
        image = Image.open(filename)
        embedding = img_model.encode(image)
        embedding_array.append(embedding)
    input_embedding = np.stack(embedding_array, 0)
    return normalize(input_embedding)

def main(n_clusters=64):
    img_model = SentenceTransformer('clip-ViT-B-32')
    rng = np.random.default_rng()
    init_embedding = 'k-means++'

    filenames = glob.glob('src_images/**/*.*', recursive=True)
    input_embedding = calc_embedding(filenames, img_model)
    init_filenames = glob.glob('key_images/**/*.*', recursive=True)
    if init_filenames is not None and len(init_filenames) > 0:
        init_embedding = calc_embedding(init_filenames, img_model)
        if n_clusters is not None and len(init_filenames) < n_clusters:
            rand_embedding = np.copy(input_embedding)
            rng.shuffle(rand_embedding)
            rand_embedding = rand_embedding[:n_clusters-len(init_filenames)]
            init_embedding = np.concatenate([init_embedding, rand_embedding], 0)

    kmeans_model = KMeans(n_clusters=n_clusters, init=init_embedding)
    kmeans_model.fit(input_embedding)

    for loop_cnt, filename in enumerate(filenames):
        print(filename + ': ' + str(kmeans_model.labels_[loop_cnt]))
        os.makedirs('dst_images/' + str(kmeans_model.labels_[loop_cnt]), exist_ok=True)
        shutil.copy2(filename, 'dst_images/' + str(kmeans_model.labels_[loop_cnt]) + '/' + filename.replace('\\', '_').replace('/', '_'))

if __name__ == '__main__':
    main()
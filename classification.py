# -*- coding: utf_8 -*-

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob

def calc_embedding(filenames, img_model):
    embedding_array = []
    for filename in tqdm(filenames, total=len(filenames)):
        image = Image.open(filename)
        embedding = img_model.encode(image)
        embedding_array.append(embedding)
    input_embedding = np.stack(embedding_array, 0)
    return normalize(input_embedding)

def main():
    img_model = SentenceTransformer('clip-ViT-B-32')

    filenames = glob.glob('src_images/**/*.*', recursive=True)
    input_embedding = calc_embedding(filenames, img_model)
    init_filenames = glob.glob('key_images/**/*.*', recursive=True)
    init_embedding = calc_embedding(init_filenames, img_model)

    kmeans_model = KMeans(n_clusters=2, init=init_embedding)
    kmeans_model.fit(input_embedding)

    for loop_cnt, filename in enumerate(filenames):
        print(filename + ': ' + str(kmeans_model.labels_[loop_cnt]))

if __name__ == '__main__':
    main()
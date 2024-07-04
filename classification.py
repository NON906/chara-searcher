# -*- coding: utf_8 -*-

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob

def main():
    img_model = SentenceTransformer('clip-ViT-B-32')

    filenames = glob.glob('src_images/**/*.*', recursive=True)
    embedding_array = []
    for filename in tqdm(filenames, total=len(filenames)):
        image = Image.open(filename)
        embedding = img_model.encode(image)
        embedding_array.append(embedding)
    input_embedding = np.stack(embedding_array, 0)

    kmeans_model = KMeans(n_clusters=2)
    kmeans_model.fit(input_embedding)

    for loop_cnt, filename in enumerate(filenames):
        print(filename + ': ' + str(kmeans_model.labels_[loop_cnt]))

if __name__ == '__main__':
    main()
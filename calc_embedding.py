# -*- coding: utf_8 -*-

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import os
import shutil
import json
from pathlib import Path
from contextlib import redirect_stderr

def calc_embedding(filenames, img_model):
    embedding_array = []
    sizes = []
    for filename in tqdm(filenames, total=len(filenames)):
        image = Image.open(filename)
        sizes.append(image.size)
        with open(os.devnull, 'w') as f:
            with redirect_stderr(f):
                embedding = img_model.encode(image)
        embedding_array.append(embedding)
    input_embedding = np.stack(embedding_array, 0)
    return input_embedding, sizes

def calc_embedding_main(dir_path='src_images'):
    img_model = SentenceTransformer('clip-ViT-B-32')
    save_npz_path = os.path.join(dir_path, 'embedding.npz')
    save_json_path = os.path.join(dir_path, 'embedding.json')

    filenames = glob.glob(os.path.join(dir_path, '**/*.*'), recursive=True)
    exclusion_files = glob.glob(os.path.join(dir_path, '**/*.txt'), recursive=True)
    exclusion_files.append(save_npz_path)
    exclusion_files.append(save_json_path)
    filenames = [filename for filename in filenames if not filename in exclusion_files]

    input_embedding, sizes = calc_embedding(filenames, img_model)

    np.savez_compressed(save_npz_path, embedding=input_embedding)

    p_abs = Path(dir_path)
    relative_file_paths = [str(Path(filename).relative_to(p_abs)) for filename in filenames]
    files_array = []
    for loop in range(len(relative_file_paths)):
        files_array.append({
            'path': relative_file_paths[loop],
            'width': sizes[loop][0],
            'height': sizes[loop][1]
        })
    with open(save_json_path, 'w') as f:
        json_dict = {'files': files_array}
        json.dump(json_dict, f)

if __name__ == '__main__':
    calc_embedding_main()
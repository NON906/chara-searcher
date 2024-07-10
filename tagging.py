# -*- coding: utf_8 -*-

import sys
import os
from typing import Generator, Iterable, Dict, List
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import re

sys.path.append(os.path.join(os.path.dirname(__file__), 'wd14-tagger-standalone'))
from tagger.interrogator import Interrogator
from tagger.interrogators import interrogators
sys.path = sys.path[:-1]

class Args:
    dir: str
    threshold: float = 0.35
    ext: str = '.txt'
    cpu: bool = False
    rawtag: bool = False
    recursive: bool = False
    exclude_tags: str = ''
    model: str = 'wd-v1-4-moat-tagger.v2'

args = Args()

interrogator = None

tag_escape_pattern = re.compile(r'([\\()])')

def postprocess_tags(
    tags: Dict[str, float],
    threshold=0.35,
    additional_tags: List[str] = [],
    exclude_tags: Iterable[str] = [],
    sort_by_alphabetical_order=False,
    add_confident_as_weight=False,
    replace_underscore=False,
    replace_underscore_excludes: List[str] = [],
    escape_tag=False
) -> Dict[str, float]:
    for t in additional_tags:
        tags[t] = 1.0

    # those lines are totally not "pythonic" but looks better to me
    tags = {
        t: c

        # sort by tag name or confident
        for t, c in sorted(
            tags.items(),
            key=lambda i: i[0 if sort_by_alphabetical_order else 1],
            reverse=not sort_by_alphabetical_order
        )

        # filter tags
        if (
            c >= threshold
            and t not in exclude_tags
        )
    }

    new_tags = []
    for tag in list(tags):
        new_tag = tag

        if replace_underscore and tag not in replace_underscore_excludes:
            new_tag = new_tag.replace('_', ' ')

        if escape_tag:
            new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

        if add_confident_as_weight:
            new_tag = f'({new_tag}:{tags[tag]})'

        new_tags.append((new_tag, tags[tag]))
    tags = dict(new_tags)

    return tags

def parse_exclude_tags() -> set[str]:
    if args.exclude_tags is None:
        return set()

    tags = []
    for str in args.exclude_tags:
        for tag in str.split(','):
            tags.append(tag.strip())

    # reverse escape (nai tag to danbooru tag)
    reverse_escaped_tags = []
    for tag in tags:
        tag = tag.replace(' ', '_').replace('\(', '(').replace('\)', ')')
        reverse_escaped_tags.append(tag)
    return set([*tags, *reverse_escaped_tags])  # reduce duplicates

def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str]) -> dict[str, float]:
    """
    Predictions from a image path
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)

    return postprocess_tags(
        result[1],
        threshold=args.threshold,
        escape_tag=tag_escape,
        replace_underscore=tag_escape,
        exclude_tags=exclude_tags)

def explore_image_files(folder_path: Path) -> Generator[Path, None, None]:
    """
    Explore files by folder path
    """
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
            yield path
        elif args.recursive and path.is_dir():
            yield from explore_image_files(path)

def tagging_main(dir='src_images', threshold=0.35):
    global args, interrogator
    args = Args()
    args.dir = dir
    args.threshold = threshold

    # get interrogator configs
    interrogator = interrogators[args.model]

    if args.cpu:
        interrogator.use_cpu()

    root_path = Path(args.dir)

    total = 0
    for image_path in explore_image_files(root_path):
        total += 1

    for image_path in tqdm(explore_image_files(root_path), total=total):
        caption_path = image_path.parent / f'{image_path.stem}{args.ext}'

        #print('processing:', image_path)
        tags = image_interrogate(image_path, not args.rawtag, parse_exclude_tags())

        tags_str = ', '.join(tags.keys())

        with open(caption_path, 'w') as fp:
            fp.write(tags_str)

if __name__ == '__main__':
    tagging_main()
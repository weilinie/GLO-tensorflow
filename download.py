from __future__ import print_function

import argparse
import os
import tarfile
import zipfile
from tqdm import tqdm
import requests


def prepare_download_dir(dir = 'data'):
    if not os.path.exists(dir):
        os.mkdir(dir)


# check, if file exists, make link
def check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)
        os.symlink(rel_link, link_file)


def download_CelebA(base_path):
    data_path = os.path.join(base_path, 'CelebA')
    images_path = os.path.join(data_path, 'images')
    if os.path.exists(data_path):
        print('[!] Found CelebA - skip')
        return

    # use requests to download dataset
    filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(base_path, filename)
    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': drive_id}, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            params = {'id': drive_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=24*1024), total=total_size, desc=save_path):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    # unzip the downloaded archive
    with zipfile.ZipFile(save_path) as zf:
        print('Extracting files from achives...')
        zf.extractall(base_path)
        zf.close()
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    os.rename(os.path.join(base_path, "img_align_celeba"), images_path)
    # os.remove(save_path)

    # add splits for train, validation and test
    train_dir = os.path.join(data_path, 'splits', 'train')
    valid_dir = os.path.join(data_path, 'splits', 'valid')
    test_dir = os.path.join(data_path, 'splits', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # these constants based on the standard CelebA splits
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637

    for i in range(0, TRAIN_STOP):
        basename = "{:06d}.jpg".format(i + 1)
        check_link(images_path, basename, train_dir)
    for i in range(TRAIN_STOP, VALID_STOP):
        basename = "{:06d}.jpg".format(i + 1)
        check_link(images_path, basename, valid_dir)
    for i in range(VALID_STOP, NUM_EXAMPLES):
        basename = "{:06d}.jpg".format(i + 1)
        check_link(images_path, basename, test_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--data_dir', type=str, default='../data')
    args= parser.parse_args()
    prepare_download_dir()
    if args.dataset == 'CelebA':
        download_CelebA(args.data_dir)
    else:
        raise Exception("[Caution! Cannot download dataset: {}]".format(args.dataset))

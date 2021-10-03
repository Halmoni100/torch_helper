import os
import shutil
import math
import random
from subprocess import Popen
import pathlib

import utilities

def rm_and_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def run_bash_script(script_str, temp_dir):
    script_str_lines = script_str.split('\n')
    script_str_lines_processed = [line.strip() for line in script_str_lines]
    script_str_processed = '\n'.join(script_str_lines_processed)

    utilities.mkdir_if_not_exist(temp_dir)

    script_file_path = os.path.join(temp_dir, 'script.sh')
    f = open(script_file_path, 'w')
    f.write('#!/usr/bin/env bash\n')
    f.write(script_str_processed)
    f.close()
    os.chmod(script_file_path, 0b111111101)

    proc = Popen(script_file_path)
    proc.wait()

def split_data(source, dest, ratio=0.2, copy=True):
    if not os.path.exists(source):
        raise ValueError("Source directory does not exist")

    categories = [x for x in os.listdir(source) if os.path.isdir(os.path.join(source, x))]

    split_data_dir = "{}_split".format(os.path.basename(source)) if dest is None else dest
    pathlib.Path(split_data_dir).mkdir(parents=True, exist_ok=True)
    train_dir = os.path.join(split_data_dir, "train")
    val_dir = os.path.join(split_data_dir, "val")
    os.mkdir(train_dir)
    os.mkdir(val_dir)

    for name in categories:
        path = os.path.join(source, name)
        new_train_path = os.path.join(train_dir, name)
        new_val_path = os.path.join(val_dir, name)
        os.mkdir(new_train_path)
        os.mkdir(new_val_path)
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        indices = list(range(len(files)))
        random.shuffle(indices)
        train_end_idx = math.ceil(len(files) * (1-ratio))
        train_indices = indices[:train_end_idx]
        val_indices = indices[train_end_idx:]
        transfer_fn = shutil.copy if copy else shutil.move
        for idx in train_indices:
            f = files[idx]
            f_path = os.path.join(path, f)
            transfer_fn(f_path, new_train_path)
        for idx in val_indices:
            f = files[idx]
            f_path = os.path.join(path, f)
            transfer_fn(f_path, new_val_path)

def shard_data(source, destinations, shuffle=True, copy=True):
    if not os.path.exists(source):
        raise ValueError("Source directory does not exist")

    transfer_fn = shutil.copy if copy else shutil.move

    for dest in destinations:
        rm_and_mkdir(dest)
    num_shards = len(destinations)

    for root, dirs, files in os.walk(source):
        relpath = os.path.relpath(root, source)
        if shuffle:
            random.shuffle(files)
        num_per_shard = math.ceil(len(files) / num_shards)
        for i in range(num_shards):
            dest = destinations[i]
            dest_path = os.path.join(dest, relpath)
            pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)
            start_idx = i * num_per_shard
            end_idx = min((i + 1) * num_per_shard, len(files))
            for file in files[start_idx:end_idx]:
                transfer_fn(os.path.join(root, file), dest_path)

GCS_BUCKET = 'chong-ml'

def download_kaggle_competition_dataset(competition, directory, temp_dir):
    bash_script = '''
    cd $DIRECTORY
    kaggle competitions download $COMPETITION
    '''
    bash_script = bash_script.replace('$DIRECTORY', directory)
    bash_script = bash_script.replace('$COMPETITION', competition)
    run_bash_script(bash_script, temp_dir)

def unzip(zip_file, directory, temp_dir, file_pattern=''):
    bash_script = '''
    unzip -q $ZIP_FILE $FILE_PATTERN -d $DIRECTORY
    '''
    bash_script = bash_script.replace('$ZIP_FILE', zip_file)
    bash_script = bash_script.replace('$FILE_PATTERN', file_pattern)
    bash_script = bash_script.replace('$DIRECTORY', directory)
    run_bash_script(bash_script, temp_dir)

# see https://stackoverflow.com/questions/47043441/what-should-i-do-about-this-gsutil-parallel-composite-upload-warning
def upload_to_gcs(file, bucket_loc, temp_dir):
    if not os.path.exists(file) or not os.path.isfile(file):
        raise RuntimeError("File does not exist or is directory")

    options = "-o GSUtil:parallel_composite_upload_threshold=150M"
    bash_script = '''
    gsutil $OPTIONS cp $FILE gs://$GCS_BUCKET/$BUCKET_LOC
    '''
    bash_script = bash_script.replace('$OPTIONS', options)
    bash_script = bash_script.replace('$FILE', file)
    bash_script = bash_script.replace('$GCS_BUCKET', GCS_BUCKET)
    bash_script = bash_script.replace('$BUCKET_LOC', bucket_loc)
    run_bash_script(bash_script, temp_dir)

class GCSDownloadExists(Exception):
    pass

# see https://medium.com/@duhroach/gcs-read-performance-of-large-files-bd53cfca4410
def download_from_gcs(bucket_loc, dest, temp_dir):
    if os.path.exists(os.path.join(dest, os.path.basename(bucket_loc))):
        raise GCSDownloadExists("File already exists in destination")

    options = "-o GSUtil:parallel_thread_count=1 -o GSUtil:sliced_object_download_max_components=$(nproc)"
    bash_script = '''
    gsutil $OPTIONS cp gs://$GCS_BUCKET/$BUCKET_LOC $DEST
    '''
    bash_script = bash_script.replace('$OPTIONS', options)
    bash_script = bash_script.replace('$DEST', dest)
    bash_script = bash_script.replace('$GCS_BUCKET', GCS_BUCKET)
    bash_script = bash_script.replace('$BUCKET_LOC', bucket_loc)
    run_bash_script(bash_script, temp_dir)

def tar(source, tar_path, temp_dir):
    working_dir = os.path.join(*os.path.split(source)[:-1])
    dir_to_tar = os.path.split(source)[-1]
    bash_script = '''
    cd $WORKING_DIR
    tar cf $TAR_PATH $DIR_TO_TAR
    '''
    bash_script = bash_script.replace('$WORKING_DIR', working_dir)
    bash_script = bash_script.replace('$TAR_PATH', tar_path)
    bash_script = bash_script.replace('$DIR_TO_TAR', dir_to_tar)
    run_bash_script(bash_script, temp_dir)

def untar(tar_path, dest, temp_dir):
    bash_script = '''
    tar xf $TAR_PATH -C $DEST
    '''
    bash_script = bash_script.replace('$TAR_PATH', tar_path)
    bash_script = bash_script.replace('$DEST', dest)
    run_bash_script(bash_script, temp_dir)

def move(source, dest, temp_dir):
    bash_script = '''
    mv $SOURCE $DEST
    '''
    bash_script = bash_script.replace('$SOURCE', source)
    bash_script = bash_script.replace('$DEST', dest)
    run_bash_script(bash_script, temp_dir)

def mkdir(path, temp_dir, parents=True):
    options = '-p' if parents else ''
    bash_script = '''
    mkdir $OPTIONS $PATH
    '''
    bash_script = bash_script.replace('$OPTIONS', options)
    bash_script = bash_script.replace('$PATH', path)
    run_bash_script(bash_script, temp_dir)

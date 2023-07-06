"""
Used to compress videos (FPS and dimensions) in the Singularity project.
Modified from https://github.com/ArrowLuo/CLIP4Clip
"""
import os
from os.path import exists, join
import argparse
import subprocess
from multiprocessing import Pool
import shutil
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
from functools import partial
from tqdm import tqdm
from PIL import Image


def resize_image(input_path, output_path, size=224):
    with Image.open(input_path) as img:
        w, h = img.width, img.height
        r = 1. * w / h
        if w > h:
            h = size
            w = r * size
        else:
            h = size / r
            w = size

        img_resized = img.resize((int(w), int(h)))
        img_resized.save(output_path)


def _compress_images(input_output_pair, size=224):
    """
    Scale and downsample an input image to a given fps and size (shorter side size).
    This also removes the audio from the image.
    """
    input_image_path, output_image_path = input_output_pair
    try:
        resize_image(input_image_path, output_image_path, size)
    except Exception as e:
        print(f"Caught Exception {e}")

def _compress_audios(input_output_pair, sample_rate=48000, replace_dir_name=None):
    """
    Scale and downsample an input video to a given fps and size (shorter side size).
    This also removes the audio from the video.
    """
    input_file_path, output_file_path = input_output_pair
    if replace_dir_name is not None:
        output_file_path = input_file_path.replace(replace_dir_name, replace_dir_name+'_processed')
        output_file_dir = '/'.join(output_file_path.split('/')[:-1])
        os.makedirs(output_file_dir, exist_ok=True)
    try:

        command = ['ffmpeg',
                '-y',  # (optional) overwrite output file if it exists
                '-i', input_file_path,
                '-ar', str(sample_rate),
                '-ac', '1',
                output_file_path, # .mp3
                ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        raise e
    

def _compress_videos(input_output_pair, size=224, fps=3, replace_dir_name=None, keep_audio=False, only_audio=False):
    """
    Scale and downsample an input video to a given fps and size (shorter side size).
    This also removes the audio from the video.
    """
    input_file_path, output_file_path = input_output_pair
    if replace_dir_name is not None:
        output_file_path = input_file_path.replace(replace_dir_name, replace_dir_name+'_processed')
        output_file_dir = '/'.join(output_file_path.split('/')[:-1])
        os.makedirs(output_file_dir, exist_ok=True)
    try:
        if only_audio:
            # command = ['ffmpeg',
            #         '-y',  # (optional) overwrite output file if it exists
            #         '-i', input_file_path,
            #         '-q:a', '0',  
            #         '-map', 'a',  # only audio
            #         output_file_path, # .mp3
            #         ]
            command = ['ffmpeg',
                    '-y',  # (optional) overwrite output file if it exists
                    '-i', input_file_path,
                    output_file_path, # .mp3
                    ]
        elif keep_audio:
            
            command = ['ffmpeg',
                    '-y',  # (optional) overwrite output file if it exists
                    '-i', input_file_path,
                    '-filter:v',  # no audio
                    f"scale='if(gt(a,1),trunc(oh*a/2)*2,{size})':'if(gt(a,1),{size},trunc(ow*a/2)*2)'",
                    # '-map', '0:v',  # no audio
                    '-r', str(fps),  # frames per second
                    # '-g', str(16),
                    output_file_path,
                    ]
        else:
            command = ['ffmpeg',
                    '-y',  # (optional) overwrite output file if it exists
                    '-i', input_file_path,
                    '-filter:v',  # no audio
                    f"scale='if(gt(a,1),trunc(oh*a/2)*2,{size})':'if(gt(a,1),{size},trunc(ow*a/2)*2)'",
                    '-map', '0:v',  # no audio
                    '-r', str(fps),  # frames per second
                    # '-g', str(16),
                    output_file_path,
                    ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        raise e


def _compress(input_output_pair, fps=3, size=224, file_type="image", replace_dir_name=None, keep_audio=False, 
              only_audio=False, sample_rate=48000):
    if file_type == "image":
        _compress_images(input_output_pair, size, replace_dir_name=replace_dir_name)
    elif file_type == "video":
        _compress_videos(input_output_pair, size, fps, replace_dir_name=replace_dir_name, keep_audio=keep_audio, only_audio=only_audio)
    elif file_type == "audio":
        _compress_audios(input_output_pair, sample_rate=sample_rate, replace_dir_name=replace_dir_name)


def prepare_input_output_pairs(input_root, output_root, input_file_list_path=None, replace_dir_name=None, 
                               only_audio=False, format=None):
    # filename list in `input_file_list_path` can be created very fast using `ls -U . >> ../video_filenames.txt`
    with open(input_file_list_path, "r") as f:
        filenames = [s.strip() for s in f.readlines()]
    print(f"There are {len(filenames)} video/images files loaded from list.")
    input_file_path_list = []
    output_file_path_list = []
    # filenames = filenames[:100]
    for e in tqdm(filenames, desc="find un-processed videos/images"):
        e = e.replace('./', '')
        input_file_path = join(input_root, e)
        # new_e = e.split('/')[-1]
        new_e = e
        output_file_path = join(output_root, new_e)
        file_dir = '/'.join(output_file_path.split('/')[:-1])
        os.makedirs(file_dir, exist_ok=True)
        
        vidformat = input_file_path.split('/')[-1].split('.')[-1]
        if format is not None:
            output_file_path = output_file_path.replace(vidformat, format)
        elif only_audio:
            output_file_path = output_file_path.replace(vidformat, 'wav')
        if replace_dir_name is not None:
            output_file_path = input_file_path.replace(replace_dir_name, replace_dir_name+'_processed')
        # print(join(input_root, e), join(output_root, new_e), e)
        if not exists(output_file_path):
            input_file_path_list.append(input_file_path)
            output_file_path_list.append(output_file_path)
    return input_file_path_list, output_file_path_list


def run_compress():
    parser = argparse.ArgumentParser(description="Compress videos/images for speed-up")
    parser.add_argument("--input_root", type=str, help="input root", required=True)
    parser.add_argument("--input_file_list_path", type=str, required=True, default=None,
                        help="list of video filenames under args.input_root, it can be "
                             "created efficiently with `ls -U /path/to/video >> /path/to/video_filenames.txt`")
    parser.add_argument("--output_root", type=str, help="output root", required=True)
    parser.add_argument("--size", type=int, default=224, help="shorter side size, aspect ratio is kept")
    parser.add_argument("--num_workers", type=int, default=24, help="#workers")
    parser.add_argument("--fps", type=int, default=3, help="fps for output video, ignored if file_type == image")
    parser.add_argument("--file_type", type=str, choices=["image", "video", "audio"], help="input file type")
    parser.add_argument("--replace_dir_name", type=str, default=None, help="input file type")
    parser.add_argument("--keep_audio", action='store_true',  help="keep audio")
    parser.add_argument("--only_audio", action='store_true',  help="only audio")

    parser.add_argument("--sample_rate", type=int, default=48000, help="sample rate for audio")
    parser.add_argument("--format", type=str, default=None, help="output file format")


    
    args = parser.parse_args()

    # set paths
    input_root = args.input_root
    output_root = args.output_root
    assert input_root != output_root
    if not exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    # prepare and find un-processed
    input_file_path_list, output_file_path_list = prepare_input_output_pairs(
        input_root, output_root, input_file_list_path=args.input_file_list_path,replace_dir_name=args.replace_dir_name,
        only_audio=args.only_audio, format=args.format
    )
    print(f"input_file_path_list[:3] {input_file_path_list[:3]}")
    print(f"output_file_path_list[:3] {output_file_path_list[:3]}")
    print("Total videos/images need to process: {}".format(len(input_file_path_list)))

    # start parallel jobs
    num_cores = cpu_count()
    num_workers = args.num_workers
    print(f"Begin with {num_cores}-core logical processor, {num_workers} workers")
    compress = partial(_compress, fps=args.fps, size=args.size, file_type=args.file_type, 
                       replace_dir_name=args.replace_dir_name, keep_audio=args.keep_audio, only_audio=args.only_audio, 
                       sample_rate=args.sample_rate)
    input_pairs = list(zip(input_file_path_list, output_file_path_list))
    with Pool(num_workers) as pool, tqdm(total=len(input_file_path_list), desc="re-encoding videos/images") as pbar:
        for idx, _ in enumerate(pool.imap_unordered(compress, input_pairs, chunksize=32)):
            pbar.update(1)

    # copy-paste failed files
    print("Compress finished, copy-paste failed files...")
    copy_count = 0
    for input_file_path, output_file_path in zip(input_file_path_list, output_file_path_list):
        if exists(input_file_path):
            if args.replace_dir_name is not None:
                output_file_path = input_file_path.replace(args.replace_dir_name, args.replace_dir_name+'_processed')
                output_file_dir = '/'.join(output_file_path.split('/')[:-1])
                os.makedirs(output_file_dir, exist_ok=True)


            if exists(output_file_path) is False or os.path.getsize(output_file_path) < 1.:
                copy_count += 1
                shutil.copyfile(input_file_path, output_file_path)
                print("Copy and replace file: {}".format(output_file_path))
    print(f"copy_count {copy_count}")


if __name__ == "__main__":
    run_compress()

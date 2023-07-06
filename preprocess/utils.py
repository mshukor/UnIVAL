import os
import json 
# import torch
# import clip
from PIL import Image
# import sng_parser
from tqdm import tqdm 
import codecs
import numpy as np
import csv
import sys

from io import BytesIO
import base64
import pickle 

from random import shuffle

import ast
# uniq-id, image (base64 string), caption, question, answer, 
#ground-truth objects (objects appearing in the caption or question), 
#dataset name (source of the data) and task type (caption, qa or visual gronunding).


# import subprocess
from multiprocessing import Pool
# import shutil
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count

from functools import partial


def remove_special(input_string):
    final_string = ""
    for character in input_string:
        if  character == " ":
            final_string = final_string + character
        else:
            if(character.isalnum()):
                final_string = final_string + character
    return final_string

def convert_img_to_str(file_name):
    img = Image.open(file_name) # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    return base64_str

def add_new_tsv(original_tsv_path, new_tsv, output_path):
    

    tsv = []
    with open(original_tsv_path) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for line in tqdm(tsv_file):
            tsv.append(line)
    start_id = len(tsv)+1
    
    print(start_id)
    for d in tqdm(new_tsv):
        d[0] = d[0] + start_id
        tsv.append(d)
    shuffle(tsv)
    
    with open(output_path, 'w', newline='') as f_output:
        csv_output = csv.writer(f_output, delimiter='\t')

        for t in tqdm(tsv):
            csv_output.writerow(t)

    return tsv


def get_tsv_caption_data_from_json(original_data, start_id, dataset_name, task_type, convert_images=True):
    tsv_data = []
    for i, d in tqdm(enumerate(original_data)):
        caption = remove_special(d['caption'])
        img_path = d['image']
        if convert_images:
            img = convert_img_to_str(img_path)
        else:
            img_path = img_path.replace('/data/mshukor/data/', '')
            img = img_path
        t = [start_id, img, caption, '','', '', dataset_name, task_type]
        tsv_data.append(t)
        start_id+=1

    return tsv_data


def get_tsv_caption_data_from_video_json(original_data, start_id, dataset_name, task_type, convert_images=True, prefix=None):
    tsv_data = []
    for i, d in tqdm(enumerate(original_data)):
        caption = remove_special(d['caption'])
        if prefix is not None:
            img_path = os.path.join(prefix, d['video'])
            
        if convert_images:
            img = convert_img_to_str(img_path)
        else:
            img_path = img_path.replace('/data/mshukor/data/', '')
            img = img_path
        t = [start_id, img, caption, '','', '', dataset_name, task_type]
        tsv_data.append(t)
        start_id+=1

    return tsv_data

def get_tsv_vqa_data_from_json(original_data, start_id, dataset_name, task_type, image_root=None, convert_images=True):
    tsv_data = []
    for i, d in tqdm(enumerate(original_data)):
        question = remove_special(d['question'])
        img_path = d['image']
        if 'COCO_' in img_path:
            img_path = os.path.join('coco/', img_path)
            
        if image_root is not None:
            img_path = os.path.join(image_root, img_path)
            
        if convert_images:
            img = convert_img_to_str(img_path)
        else:
            img_path = img_path.replace('/data/mshukor/data/', '')
            img = img_path
            
        answers = set(d['answer'])
        
        answer_weight = {}
        for ans in answers:
            ans = remove_special(ans)
            if ans in answer_weight.keys():
                answer_weight[ans] += 1/len(answers)
            else:
                answer_weight[ans] = 1/len(answers)

        ans_ = ["{:.1f}".format(conf)+'|!+'+ans for ans, conf in answer_weight.items()]
        ans_ = '&&'.join(ans_)
        
        t = [start_id, img, '', question, ans_, '', dataset_name, task_type]
        tsv_data.append(t)
        start_id+=1
    shuffle(tsv_data)
    return tsv_data


def get_tsv_vqa_synth_data_from_json(original_data, start_id, dataset_name, task_type, image_root=None, convert_images=True, data_type='all'):
    tsv_data = []
    for i, d in tqdm(enumerate(original_data)):
        if data_type == 'manual' and 'manual' in d['dataset']:
            pass
        elif data_type == 'auto' and 'manual' not in d['dataset']:
            pass 
        elif data_type == 'all':
            pass 
        else:
            continue


        question = remove_special(d['question'])
        img_path = d['image']
        if 'COCO_' in img_path:
            img_path = os.path.join('coco/', img_path)
            
        if image_root is not None:
            img_path = os.path.join(image_root, img_path)
            
        if convert_images:
            img = convert_img_to_str(img_path)
        else:
            img_path = img_path.replace('/data/mshukor/data/', '')
            img = img_path
            
        answers = set(d['answer'])
        
        answer_weight = {}
        for ans in answers:
            ans = remove_special(ans)
            if ans in answer_weight.keys():
                answer_weight[ans] += 1/len(answers)
            else:
                answer_weight[ans] = 1/len(answers)

        ans_ = ["{:.1f}".format(conf)+'|!+'+ans for ans, conf in answer_weight.items()]
        ans_ = '&&'.join(ans_)
        
        t = [start_id, img, '', question, ans_, '', dataset_name, task_type]
        tsv_data.append(t)
        start_id+=1
    shuffle(tsv_data)
    return tsv_data

def get_tsv_from_vg_grounding(regions, data, start_id, dataset_name='visual_genome', task_type='visual_grounding', convert_images=True, split='train', thresh=16384):
    
    original_data = json.load(open(regions,'r'))
    
    image_data = json.load(open(data,'r'))


    
    id_2_imagepath = {}

    for d in tqdm(image_data):
        id_ = int(d['image'].split('/')[-1].split('.')[0])
        id_2_imagepath[id_] = d['image']
    
    tsv_data = []
    for d in tqdm(original_data):
        img_path = id_2_imagepath[d['id']]
        if convert_images:
            img = convert_img_to_str(img_path)
        else:
            img_path = img_path.replace('/data/mshukor/data/', '')
            img = img_path
            
        for reg in d['regions']:
            width = reg['width']
            height = reg['height']
            x = reg['x']
            y = reg['y']
            area = width*height
            if area < thresh:
                x1, y1, x2, y2 = x, y, x + width, y + height   # top left, bottom right 
                box = '{:.2f},{:.2f},{:.2f},{:.2f}'.format(x1, y1, x2, y2)
                sentence = remove_special(reg['phrase'])
                t = [start_id, img, sentence, '', box, '', dataset_name, task_type]
                tsv_data.append(t)
                start_id+=1
    shuffle(tsv_data)
    return tsv_data

def get_tsv_from_refcoco(ref_path, instances_path, start_id, dataset_name='refcoco_train', task_type='visual_grounding', convert_images=True, split='train'):
    
    refs = pickle.load(open(ref_path, 'rb'))
    instances = json.load(open(instances_path,'r'))
    
    id_to_annot = {}
    for annot in tqdm(instances['annotations']):
        id_to_annot[annot['id']] = annot
        
    id_to_images = {}
    for annot in tqdm(instances['images']):
        id_to_images[annot['id']] = annot
    
    tsv_data = []
    for ref in tqdm(refs):
        ref_split = ref['split']
        if ref_split == split:
            image_id = ref['image_id']
            file_name = id_to_images[ref['image_id']]['file_name']
            if ref_split == 'train':
                file_name = os.path.join('coco/train2014', file_name)
                
            if convert_images:
                img_path = os.path.join('/data/mshukor/data/', file_name)
                img = convert_img_to_str(img_path)
            else:
                img_path = file_name.replace('/data/mshukor/data/', '')
                img = img_path

            ann_id = ref['ann_id']
            annot = id_to_annot[ann_id]
            bbox = annot['bbox'] # x,y,w,h bottom left
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]   # top left, bottom right 
            box = '{:.2f},{:.2f},{:.2f},{:.2f}'.format(x1, y1, x2, y2)
            for sent in ref['sentences']:
                sentence = remove_special(sent['sent'])
                # [id, image, 'third book starting from left', '', '29.1,11.72,66.81,343.41', '', 'refcoco_train', 'visual_grounding']
                t = [start_id, img, sentence, '', box, '', dataset_name, task_type]

                tsv_data.append(t)
                start_id+=1
    shuffle(tsv_data)
    return tsv_data

def get_tsv_data_from_jsons(datasets, start_id, task_types, image_root=None, convert_images=True):
    tsvs = []
    for (original_data_path, task_type) in zip(datasets, task_types):
        print(task_type)
        if task_type == 'caption':
            dataset_name = original_data_path.split('/')[-1].split('.')[0]
            print(dataset_name,'start_id:', start_id)
            original_data = json.load(open(original_data_path,'r'))
            tsvs += get_tsv_caption_data_from_json(original_data=original_data, start_id=start_id, dataset_name=dataset_name, task_type=task_type, convert_images=convert_images)
        elif task_type == 'qa':
            dataset_name = original_data_path.split('/')[-1].split('.')[0]
            print(dataset_name,'start_id:', start_id)
            original_data = json.load(open(original_data_path,'r'))
            tsvs += get_tsv_vqa_data_from_json(original_data=original_data, start_id=start_id, dataset_name=dataset_name, task_type=task_type, image_root=image_root, convert_images=convert_images)

        elif task_type == 'visual_grounding':
            dataset_name = original_data_path[0].split('/')[-2].replace('+', '')+'_train'
            print(dataset_name,'start_id:', start_id)
            if dataset_name == 'refcoco_train':
                tsvs += get_tsv_from_refcoco(original_data_path[0], original_data_path[1], start_id, dataset_name=dataset_name, task_type=task_type, convert_images=convert_images, split='train')
                
        elif task_type == 'detection':
            dataset_name = original_data_path[0]
            if dataset_name == 'vg':
                tsvs+= get_tsv_from_vg_detection(original_data_path[1], original_data_path[2], start_id, convert_images=convert_images, split='train')
            elif dataset_name == 'coco':
                tsvs+= get_tsv_from_coco_detection(original_data_path[1], start_id, convert_images=convert_images, split='train')
            
        else:
            raise
        start_id = tsvs[-1][0] + 1
    shuffle(tsvs)
    return tsvs



def create_imagenet_txt_files(path_data, output_path, dataset='imagenet'):
    data = []
    # start_id = 0
    for root, dirs, files, in os.walk(path_data):
        for d in tqdm(dirs):
            dir_path = os.path.join(root, d)
            for _, _, dir_files in os.walk(dir_path):
                for f in dir_files:
                    file_path = os.path.join(dir_path, f)
                    if dataset == 'imagenet21k':
                        file_path = '/'.join(file_path.split('/')[-3:])
                    elif dataset == 'openimages':
                        file_path = '/'.join(file_path.split('/')[-4:])
                    elif dataset == 'yfcc':
                        file_path = '/'.join(file_path.split('/')[-5:])
                    elif dataset == 'imagenet':
                        file_path = '/'.join(file_path.split('/')[-5:])
                    else:
                        file_path = '/'.join(file_path.split('/')[-4:])
                    image_id = f.split('.')[0]
                    tmp = [image_id, file_path]
                    data.append(tmp)
                    # start_id+=1

    with open(output_path, 'w', newline='') as f_output:
        csv_output = csv.writer(f_output, delimiter='\t')

        for t in tqdm(data):
            csv_output.writerow(t)
            
            
            
def get_tsv_from_vg_detection(instances_path, path_images, start_id, convert_images=True, split='train'):
    print('start id:', start_id)
    instances = json.load(open(instances_path,'r'))
    
    id_to_objects = {}
    for d in instances:
        id_to_objects[d['id']] = d


    
    id_to_image_path = {}
    for root, dirs, files, in os.walk(path_images):
        for d in dirs:
            dir_path = os.path.join(root, d)
            for _, _, dir_files in os.walk(dir_path):
                for f in dir_files:
                    file_path = os.path.join(dir_path, f)
                    file_path = '/'.join(file_path.split('/')[-4:])
                    image_id = f.split('.')[0]
                    id_to_image_path[image_id] = file_path

                    


    tsv_data = []
    missied = []
    for ref in tqdm(id_to_image_path.keys()):
        ref_split = split
        
        image_id = ref
        
        file_name = id_to_image_path[image_id]
        if convert_images:
            img_path = os.path.join('/data/mshukor/data/', file_name)
            img = convert_img_to_str(img_path)
        else:
            img_path = file_name.replace('/data/mshukor/data/', '')
            img = img_path

            
        if int(image_id) in id_to_objects:
            objects = id_to_objects[int(image_id)]['objects']
        else:
            missied.append(image_id)
            continue
        
        if len(objects) == 0:
            missied.append(image_id)
            continue
            
        
        areas = []
        detections = []
        for annot in objects:
            x,y,w,h = annot['x'], annot['y'], annot['w'], annot['h'] # x,y,w,h bottom left
            
            area = w*h
            
            x1, y1, x2, y2 = x, y, x + w, y + h  # top left, bottom right 
            
            x1 = max(0, x1)
            x2 = max(0, x2)
            
            
            category = ','.join(remove_special(annot['names'])).replace('\x00','')
            object_id = annot['id']
            
            
            tmp = '{:.3f},{:.3f},{:.3f},{:.3f},{},{}'.format(x1, y1, x2, y2, object_id, category)
            detections.append(tmp)
            areas.append(area)

        sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
        detections = [detections[k] for k in sorted_indices]
        
        detections = '&&'.join(detections)
        t = [start_id, img, detections]

        tsv_data.append(t)
        start_id+=1
    print('missed images:', len(missied))
    shuffle(tsv_data)
    return tsv_data



def get_tsv_from_coco_detection(instances_path, start_id, convert_images=True, split='train'):
    print('start id:', start_id)
    instances = json.load(open(instances_path,'r'))
    imgid_to_annot = {}
    for annot in tqdm(instances['annotations']):
        if annot['image_id'] not in imgid_to_annot:
            imgid_to_annot[annot['image_id']] = [annot]
        else:
            imgid_to_annot[annot['image_id']].append(annot)

    id_to_category = {}
    for annot in tqdm(instances['categories']):
        id_to_category[annot['id']] = annot['name']

    tsv_data = []
    missied = []
    for ref in tqdm(instances['images']):
        ref_split = split
        image_id = ref['id']
        file_name = ref['file_name']

        if ref_split == 'train':
            file_name = os.path.join('coco/train2014', file_name)

        if convert_images:
            img_path = os.path.join('/data/mshukor/data/', file_name)
            img = convert_img_to_str(img_path)
        else:
            img_path = file_name.replace('/data/mshukor/data/', '')
            img = img_path

        # ann_id = ref['id']
        # annot = id_to_annot[ann_id]
        if image_id not in imgid_to_annot:
            missied.append(image_id)
            continue
        annots = imgid_to_annot[image_id]
        detections = []
        areas = []
        for annot in annots:
            bbox = annot['bbox'] # x,y,w,h bottom left
            area = bbox[2]*bbox[3]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]   # top left, bottom right 
            # box = '{:.3f},{:.3f},{:.3f},{:.3f}'.format(x1, y1, x2, y2)

            object_id = annot['category_id']
            category = remove_special(id_to_category[object_id])

            tmp = '{:.3f},{:.3f},{:.3f},{:.3f},{},{}'.format(x1, y1, x2, y2, object_id, category)
            areas.append(area)
            detections.append(tmp)

        sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
        detections = [detections[k] for k in sorted_indices]
        detections = '&&'.join(detections)
        t = [start_id, img, detections]

        tsv_data.append(t)
        start_id+=1
    shuffle(tsv_data)
    return tsv_data

def imagepath_to_image_size(img_path, dir_path):
    img_path = os.path.join(dir_path, img_path)
    w, h = Image.open(img_path).size
    # imageid_to_meta_dict[img_path] = [w, h]
    return w, h, img_path


def save_imageid_to_meta_dict(path_images, output_path, mp=False, num_workers=1):

    id_to_image_path = {}
    for file in os.listdir(path_images):
        file_path = os.path.join(path_images, file)
        file_path = '/'.join(file_path.split('/')[-4:])
        image_id = file.split('.')[0]
        id_to_image_path[image_id] = file

    imageid_to_meta_dict = {}

    if mp:
        iterable = list(id_to_image_path.values())
        mp_func = partial(imagepath_to_image_size, dir_path=path_images,)

        num_cores = cpu_count()
        num_workers = num_workers
        print(f"Begin with {num_cores}-core logical processor, {num_workers} workers")
        with Pool(num_workers) as pool, tqdm(total=len(iterable), desc="running") as pbar:
            for idx, res in enumerate(pool.imap_unordered(mp_func, iterable, chunksize=32)):
                
                w, h, img_path = res  
                
                imageid_to_meta_dict[img_path] = [w, h]
                pbar.update(1)
    else:
        for k, p in tqdm(id_to_image_path.items()):


            w, h, img_path = imagepath_to_image_size(path_images, p)

            imageid_to_meta_dict[img_path] = [w, h]


    print(len(imageid_to_meta_dict))
    with open(output_path, 'w') as f:
        json.dump(imageid_to_meta_dict, f)

    return imageid_to_meta_dict


def get_tsv_from_openimages_detection(instances_path, path_images, class_path, 
    start_id, convert_images=False, split='train', image_root='/gpfsdswork/dataset', image_meta=None):

    id_to_image_path = {}
    # for root, dirs, files, in os.walk(path_images):
    #     for d in dirs:
    #         dir_path = os.path.join(root, d)
    #         print(dir_path)
    #         for _, _, dir_files in os.walk(dir_path):
    #             for f in dir_files:
    #                 print(f)
    #                 file_path = os.path.join(dir_path, f)
    #                 file_path = '/'.join(file_path.split('/')[-4:])
    #                 image_id = f.split('.')[0]
    #                 id_to_image_path[image_id] = file_path

    for file in os.listdir(path_images):
        file_path = os.path.join(path_images, file)
        file_path = '/'.join(file_path.split('/')[-4:])
        image_id = file.split('.')[0]
        id_to_image_path[image_id] = file

    def imagepath_to_image_size(img_path):
        w, h = Image.open(img_path).size
        return w, h

    id_to_annot = {}
    with open(instances_path) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for i, line in tqdm(enumerate(tsv_file)):
            if i == 0:
                continue # skip header
            img_id = line[0].split(',')[0]
            if img_id in id_to_annot:
                id_to_annot[img_id].append(line)
            else:
                id_to_annot[img_id] = [line]

    classid_to_class = {}

    with open(class_path) as file:
        tsv_file = csv.reader(file, delimiter=',')
        for i, line in tqdm(enumerate(tsv_file)):
            classid_to_class[line[0]] = line[1]

    if image_meta is not None:
        image_size = json.load(open(image_meta, 'r'))
    else:
        image_size = None 

    tsv_data = []
    for i, img_id in tqdm(enumerate(id_to_annot.keys())):
        annots = id_to_annot[img_id]
        if img_id in id_to_image_path:
            img_path = id_to_image_path[img_id]
            orig_img_path = os.path.join(path_images, img_path)

            save_img_path = os.path.join(image_root, img_path)

            if image_size is None:
                w, h = imagepath_to_image_size(orig_img_path)
            else:
                w, h = image_size[orig_img_path]

            if convert_images:
                img = convert_img_to_str(orig_img_path)
            else:
                img = save_img_path

            areas = []
            detections = []
            for d in annots:
                d = d[0].split(',')

                x1, x2, y1, y2 = d[4:8]
                x1, x2, y1, y2 = float(x1), float(x2), float(y1), float(y2)

                x1, x2, y1, y2 = x1*w, x2*w, y1*h, y2*h
                box_w, box_h = x2 - x1, y2 - y1
                area = box_w*box_h
                areas.append(area)

                object_id = d[2]
                category = remove_special(classid_to_class[object_id])

                tmp = '{:.3f},{:.3f},{:.3f},{:.3f},{},{}'.format(x1, y1, x2, y2, object_id, category)
                detections.append(tmp)


            sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
            detections = [detections[k] for k in sorted_indices]

            detections = '&&'.join(detections)
            t = [start_id, img, detections]

            tsv_data.append(t)
            start_id+=1

    shuffle(tsv_data)
    return tsv_data


    
def replace_image_id_by_path(input_tsv, output_tsv, mapping_file):
    selected_cols='0,1,2'
    data = []
    selected_col_ids = [int(col_id) for col_id in selected_cols.split(",")]
    with open(input_tsv) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for line in tqdm(tsv_file):
            d = [line[i] for i in selected_col_ids]
            data.append(d)
            
    im_id_to_path = {}
    with open(mapping_file) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for line in tqdm(tsv_file):
            d = [line[i] for i in [0, 1]]
            im_id_to_path[d[0]] = d[1]
            
    for d in tqdm(data):
        im_id = d[1].split('/')[-1].split('.')[0]
        im_path = im_id_to_path[im_id]
        d[1] = im_path
        
    with open(output_tsv, 'w', newline='') as f_output:
        csv_output = csv.writer(f_output, delimiter='\t')

        for t in tqdm(data):
            csv_output.writerow(t)
        
    return data



####3 video

def get_tsv_msrvtt_vqa_data_from_json(original_data, start_id, image_root=None, convert_images=False):
    tsv_data = []
    for i, d in tqdm(enumerate(original_data)):
        question = remove_special(d['question'])+'?'
        img_path = d['video']

        img_id = img_path.split('.')[0]
        
        
        if image_root is not None:
            img_path = os.path.join(image_root, img_path)
            
        if convert_images:
            img = convert_img_to_str(img_path)
        else:
            img_path = img_path.replace('/data/mshukor/data/', '')
            img = img_path
            
        answer = remove_special(d['answer'])
        
        conf = 1.0
        
        ans = "{:.1f}".format(conf)+'|!+'+answer

        
        t = [start_id, img_id, question, ans, '', img]
        tsv_data.append(t)
        start_id+=1
    shuffle(tsv_data)
    return tsv_data



def get_tsv_msrvtt_caption_data_from_json(original_data, start_id, image_root=None, convert_images=False):
    tsv_data = []
    for i, d in tqdm(enumerate(original_data)):
        caption = d['caption']
        if isinstance(caption, list):
            cs = [remove_special(c) for c in caption]
            caption = '&&'.join(cs)
        else:
            caption = remove_special(caption)
        img_path = d['video']
        img_id = img_path.split('.')[0]
        
        if image_root is not None:
            img_path = os.path.join(image_root, img_path)
            
        if convert_images:
            img = convert_img_to_str(img_path)
        else:
            img_path = img_path.replace('/data/mshukor/data/', '')
            img = img_path
        
        t = [start_id, img_id, caption, '', img]
        
        tsv_data.append(t)
        start_id+=1

    shuffle(tsv_data)
    return tsv_data



######3 Pile


def get_tsv_from_piletext_data(path, output_path, start_id=0, num_max_characters=2500, dataset_names=None, keepspecial=False):
    print("consider only", dataset_names)

    tsv = []
    failed = 0
    with open(output_path, 'w', newline='') as f_output:
        csv_output = csv.writer(f_output, delimiter='\t')

        
        with open(path, "rb") as f:
            for d in tqdm(f):
                d_str = d.decode("UTF-8")
                d_dict = ast.literal_eval(d_str)
                data_name = d_dict['meta']['pile_set_name']
                text = d_dict['text'][:num_max_characters]

                if dataset_names is not None and data_name in dataset_names:
        
                    text = text.replace('\t', ' ').replace("\n", ' ').replace('\"', '')
                    if not keepspecial:
                        text = remove_special(text)
                    item = [start_id, text]
                    try:
                        csv_output.writerow(item)
                    except: # (UnicodeEncodeError,SyntaxError)
                        failed+=1
                        continue
                
                    start_id+=1
                    tsv.append(item)
    print("len", len(tsv), "failed", failed)
    return tsv
        
    


def save_pile_tsvs(path, output_path, dataset_names, dir_names=None, keepspecial=False, num_max_characters=1500, prefix=''):
    
    print('prepare:', dir_names)
    
    for filename in os.listdir(path):
        if dir_names is not None and filename in dir_names:
            input_path = os.path.join(path, filename)
            if 'jsonl' in filename:
                output_file_name = filename.split('.')[0]+prefix+'_pile.tsv'
                output_file_name = os.path.join(output_path, output_file_name)
                print("creating:", output_file_name, "from", input_path)
                tsv = get_tsv_from_piletext_data(input_path, output_file_name, start_id=0, num_max_characters=num_max_characters, 
                                                 dataset_names=dataset_names, keepspecial=keepspecial)
            
    return tsv


def add_pile_tsvs(path, output_path='pile_all.tsv', key='pile.tsv'):
    
    start_id = 0
    with open(output_path, 'w', newline='') as f_output:
        csv_output = csv.writer(f_output, delimiter='\t')

        for filename in os.listdir(path):
            input_path = os.path.join(path, filename)
            if key in filename:

                with open(input_path) as file:
                    tsv_file = csv.reader((line.replace('\0','') for line in file), delimiter='\t')
                    for line in tqdm(tsv_file):
                        line[0] = start_id                        
                        csv_output.writerow(line)
                        start_id+=1
            print('start id', line[0])
    # return tsv
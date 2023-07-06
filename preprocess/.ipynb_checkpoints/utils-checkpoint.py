import os
import json 
import torch
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

# uniq-id, image (base64 string), caption, question, answer, 
#ground-truth objects (objects appearing in the caption or question), 
#dataset name (source of the data) and task type (caption, qa or visual gronunding).

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


def get_tsv_vqa_data_from_json(original_data, start_id, dataset_name, task_type, image_root=None, convert_images=True):
    tsv_data = []
    for i, d in tqdm(enumerate(original_data)):
        question = remove_special(d['question'])
        img_path = d['image']
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
            if ans in answer_weight.keys():
                answer_weight[ans] += 1/len(answers)
            else:
                answer_weight[ans] = 1/len(answers)

        ans_ = ["{:.1f}".format(conf)+'|!+'+ans for ans, conf in answer_weight.items()]
        ans_ = '&&'.join(ans_)
        
        t = [start_id, img, '', question, ans_, '', dataset_name, task_type]
        tsv_data.append(t)
        start_id+=1
            
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

    return tsv_data


def get_tsv_from_openimages_detection(instances_path, path_images, class_path, 
    start_id, convert_images=False, split='train', image_root='/gpfsdswork/dataset'):

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

    tsv_data = []
    for i, img_id in tqdm(enumerate(id_to_annot.keys())):
        annots = id_to_annot[img_id]
        img_path = id_to_image_path[img_id]
        orig_img_path = os.path.join(image_root, img_path)

        w, h = imagepath_to_image_size(orig_img_path)

        if convert_images:
            img = convert_img_to_str(orig_img_path)
        else:
            img = img_path

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
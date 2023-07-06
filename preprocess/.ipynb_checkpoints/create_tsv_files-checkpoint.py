from utils import get_tsv_data_from_jsons, create_imagenet_txt_files
import csv
from io import StringIO
from tqdm import tqdm 

        
# with image conversion
# datasets = ['/data/mshukor/data/our_albef_data/json_pretrain/vg_albef.json',
#             '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json',
#             ]

# output_paths = ['/data/mshukor/data/ofa/pretrain_ours/vg_albef.tsv', 
#                '/data/mshukor/data/ofa/pretrain_ours/sbu.tsv',
#                ]

# task_types = ['caption', 
#              'caption']

# start_id = 566747
# for data, task_type, output_path in zip(datasets, task_types, output_paths):

    

#     tsvs = get_tsv_data_from_jsons([data], start_id, [task_type])

#     start_id = tsvs[-1][0] + 1
    
#     print("save tsv to:", output_path)

#     with open(output_path, 'w', newline='') as f_output:
#         csv_output = csv.writer(f_output, delimiter='\t')
#         for t in tqdm(tsvs):
#             csv_output.writerow(t)

########################################################
# without image conversion
    
# datasets = ['/data/mshukor/data/our_albef_data/json_pretrain/coco_karp.json',
#             '/data/mshukor/data/our_albef_data/json_pretrain/vg_albef.json',
#             '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json',
#             '/data/mshukor/data/our_albef_data/json_pretrain/cc3m.json']

# start_id = 0
# task_types = ['caption',
#              'caption',
#              'caption',
#              'caption']

# tsvs = get_tsv_data_from_jsons(datasets, start_id, task_types, convert_images=False)

# output_path = '/data/mshukor/data/ofa/pretrain_ours/vision_language_4m.tsv'

# with open(output_path, 'w', newline='') as f_output:
#     csv_output = csv.writer(f_output, delimiter='\t')

#     for t in tqdm(tsvs):
#         csv_output.writerow(t)
        
        
        
########################################################

        
# datasets = [
#             '/data/mshukor/data/our_albef_data/json_pretrain/coco_karp.json',
#             '/data/mshukor/data/our_albef_data/json_pretrain/vg_albef.json',
#             '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json',
#             '/data/mshukor/data/our_albef_data/json_pretrain/cc3m.json', 
    
#             ['/data/mshukor/data/refcoco/refcoco+/refs(unc).p', '/data/mshukor/data/refcoco/refcoco+/instances.json'],
            
#             '/data/mshukor/data/our_albef_data/data/vqa_train.json',
# ]

# start_id = 0
# task_types = ['caption',
#              'caption',
#              'caption',
#              'caption',
#              'visual_grounding',
#              'qa',]

# tsvs = get_tsv_data_from_jsons(datasets, start_id, task_types, convert_images=False)


# output_path = '/data/mshukor/data/ofa/pretrain_ours/vision_language_mini.tsv'

# with open(output_path, 'w', newline='') as f_output:
#     csv_output = csv.writer(f_output, delimiter='\t')

#     for t in tqdm(tsvs):
#         csv_output.writerow(t)



#### imagenet

path_data = '/data/mshukor/data/imagenet/val'
output_path = '/data/mshukor/data/ofa/pretrain_ours/imagenet_val.txt'


create_imagenet_txt_files(path_data, output_path)



####### object detection



        
from preprocess.utils import get_tsv_data_from_jsons

datasets = [
    ['coco', '/data/mshukor/data/coco/annotations/instances_train2014.json'],
    ['vg', '/data/mshukor/data/visual_genome/annotations/objects.json', '/data/mshukor/data/visual_genome/images'],
]

start_id = 0
task_types = ['detection', 
              'detection',]



tsvs = get_tsv_data_from_jsons(datasets, start_id, task_types, convert_images=False)


output_path = '/data/mshukor/data/ofa/pretrain_ours/detection_mini.tsv'

with open(output_path, 'w', newline='') as f_output:
    csv_output = csv.writer(f_output, delimiter='\t')

    for t in tqdm(tsvs):
        csv_output.writerow(t)
import torch 
import numpy as np
import os 
import re

def average(checkpoints, lambdas=[0.5, 0.5], num_models=6, output_dir=None, filename=None, skip_keys=None, ema=False):

	ckpt = torch.load(checkpoints[0], map_location='cpu')

	if ema:
		key = 'extra_state'
		state = ckpt['extra_state']['ema']
	else:
		key = 'model'
		state = ckpt['model']

	print(lambdas)


	
	if num_models == 1:
		average_state = {k : v.clone() * lambdas[0] for k, v in state.items()}
		for i in range(1, len(checkpoints)):
			skip_keys_list = set()
			print(checkpoints[i], lambdas[i])
			if ema:
				statei = torch.load(checkpoints[i], map_location='cpu')['extra_state']['ema']
			else:
				statei = torch.load(checkpoints[i], map_location='cpu')['model']
			for k, v in average_state.items():
				if k in statei and (skip_keys is None or ((not any([re.match(sk, k) for sk in skip_keys])) and (not any([sk in k for sk in skip_keys])))):
					try:
						average_state[k] += (lambdas[i])*statei[k].clone()
					except:
						print(k, average_state[k].shape, statei[k].shape)
						average_state[k] += (lambdas[i])*average_state[k].clone()
				else:
					average_state[k] += (lambdas[i])*average_state[k].clone()
					skip_keys_list.add(k)
					
					
			state_dict = average_state
			print(skip_keys_list)
		if ema:
			save_obj = {key:{'ema': state_dict, 'epoch': 0}} 
			for k, v in ckpt['extra_state'].items():
				if k != 'ema':
					save_obj['extra_state']=v
					print(k)
			for k, v in ckpt.items():
				if k != key:
					save_obj[k]=v
					print(k)
		else:
			save_obj = {key: state_dict,}
			for k, v in ckpt.items():
				if k != key:
					save_obj[k]=v
				print(k)
		output_path = os.path.join(output_dir, '{}.pt'.format(filename))
		print('saving', output_path)
		torch.save(save_obj, output_path)  

	else:
		if ema:
			state_dict1 = ckpt['extra_state']['ema']
			state_dict2 = torch.load(checkpoints[1], map_location='cpu')['extra_state']['ema']
		else:
			state_dict1 = ckpt['model']
			state_dict2 = torch.load(checkpoints[1], map_location='cpu')['model']
		for l in lambdas:
			average_state = {k : v * l for k, v in state_dict1.items()} #{k : v * (1./NUM_MODELS) for k, v in state_dict1.items()}
			for k, v in average_state.items():
				if k in state_dict2:
					average_state[k] += (1-l)*state_dict2[k]
				else:
					average_state[k] += (1-l)*state_dict1[k]

			state_dict = average_state

			if ema:
				save_obj = {key:{'ema': state_dict,}} 
				for k, v in ckpt['extra_state'].items():
					if k != 'ema':
						save_obj['extra_state'][k]=v
						print(k)
				for k, v in ckpt.items():
					if k != key:
						save_obj[k]=v
						print(k)
			else:
				save_obj = {key: state_dict,}
				for k, v in ckpt.items():
					if k != key:
						save_obj[k]=v
						print(k)
			output_path = os.path.join(output_dir, '{}_l{:.2f}.pt'.format(filename, l))
			print('saving', output_path)
			torch.save(save_obj, output_path)  






# average of several models 

# lambdas = [1/4, 1/4, 1/4, 1/4]

# num_models=1
# output_dir='/lus/scratch/NAT/gda2204/SHARED/logs/ofa/pretrained_models/average_models/'
# filename='avg_caprefsnlivqa'

# checkpoints = [
# 			'/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/caption/caption_stage_1_ofaplus_base_pretrain_s2_hsep1_bs16_shuf/10_0.06_6000/checkpoint_best.pt',
# 			'/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/refcocoplus/refcocoplus_ofaplus_base_pretrain_s2_hsep1_fix_lr5e5_bs8_4_shuf/10_5e-5_512/checkpoint_best.pt',
# 			'/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/snli_ve/snli_ve_ofaplus_base_pretrain_s2_hsep1/10_5e-5/checkpoint_best.pt',
# 			'/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/vqa/vqa_ofaplus_base_pretrain_s2_bs16_lr1e4_shuf_hsep1/20_0.04_1e-4_480/checkpoint_best.pt',
# 			]

# for weight interpolation
num_models=6
output_dir='/lus/scratch/NAT/gda2204/SHARED/logs/ofa/pretrained_models/average_models/'
filename='avg_capvqa'
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

checkpoints = ['/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/vqa/vqa_ofaplus_base_pretrain_s2_bs16_lr1e4_shuf_hsep1/20_0.04_1e-4_480/checkpoint_best.pt',
	       '/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/caption/caption_stage_1_ofaplus_base_pretrain_s2_hsep1_bs16_shuf/10_0.06_6000/checkpoint_best.pt',
    ]



average(checkpoints, lambdas=lambdas, num_models=num_models, output_dir=output_dir, filename=filename)

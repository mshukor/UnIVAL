import torch

def change_name(ckpt_path, name):
    print(ckpt_path, name)
    state = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state['cfg']['model']._name = name
    state['cfg']['model']._arch = name
    torch.save(state, ckpt_path)

name="unival_base"

# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_caption_stage_1/checkpoint_best.pt"
# change_name(ckpt_path, name)


# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_refcoco/checkpoint_best.pt"
# change_name(ckpt_path, name)
# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_refcocog/checkpoint_best.pt"
# change_name(ckpt_path, name)
# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_refcocoplus/checkpoint_best.pt"
# change_name(ckpt_path, name)

# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_s1/checkpoint15.pt"
# change_name(ckpt_path, name)
# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_s2_hs/checkpoint1.pt"
# change_name(ckpt_path, name)
# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_vqa/checkpoint_best.pt"
# change_name(ckpt_path, name)


# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_video_caption_stage_1/checkpoint_best.pt"
# change_name(ckpt_path, name)
# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_video_caption_activitynet_stage_1/checkpoint_best.pt"
# change_name(ckpt_path, name)
# ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_video_vqa/checkpoint_best.pt"
# change_name(ckpt_path, name)
ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_video_vqa_msvd/checkpoint_best.pt"
change_name(ckpt_path, name)
ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_snli_ve/checkpoint_best.pt"
change_name(ckpt_path, name)
ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_audio_caption/checkpoint_best.pt"
change_name(ckpt_path, name)
ckpt_path = "/data/mshukor/logs/ofa/best_models/unival_audio_caption_clotho/checkpoint_best.pt"
change_name(ckpt_path, name)



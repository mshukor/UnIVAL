import json
import sys
import os.path as op

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

sys.path.append("/lus/home/NAT/gda2204/mshukor/code/ofa_ours")
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD

def evaluate_on_coco_caption(res_file, label_file, outfile=None, eval_cider_cached_tokens=None):
    """
    res_file: txt file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf".
    label_file: JSON file of ground truth captions in COCO format.
    """
    # ##############################
    # print("eval with CidderD scorer...")
    # eval_cider_cached_tokens = "/lus/scratch/NAT/gda2204/SHARED/data/ofa/video_data/caption_data/cider_cached_tokens/msrvtt-test3k-words.p"
    # CiderD_scorer = CiderD(df=eval_cider_cached_tokens)

    # gts = json.load(open(label_file))['annotations']
    # res_ =json.load(open(res_file))
    # print(len(res_), len(gts))
    # # print(res_)
    # gts_ = {}
    # for i in range(len(gts)):
    #     key = gts[i]['image_id']
    #     if key in gts_:
    #         gts_[key] += [gts[i]['caption']]
    #     else:
    #         gts_[key] = [gts[i]['caption']]

    # res_ = [{'image_id': r['image_id'], 'caption': [r['caption']]} for r in res_]

    # _, scores = CiderD_scorer.compute_score(gts_, res_)
    # print(len(scores))
    # print("CIDErD: ", scores)
    # print("CIDErD: ", sum(scores) / len(scores))

    # #############################3
    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file)

    ### clean result file if theres is more than one caption for each image 
    for i, id_ in enumerate(cocoRes.getImgIds()):
        res = cocoRes.imgToAnns[id_]
        if len(res) > 1: # to fix later in the code, the model should generate one caption 
            cocoRes.imgToAnns[id_] = [res[0]]
            print("found more than one predictions: {} for img, to {}".format(res, cocoRes.imgToAnns[id_]))


    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()




    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)


    return result


if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise NotImplementedError
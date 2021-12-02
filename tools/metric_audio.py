import json
import argparse
import numpy as np 
fs = 48000

def dec2sam(dec_time, fs = 48000):
    min = int(dec_time.split(':')[0])
    sec = int(dec_time.split(':')[1])
    sec += min * 60
    return sec * fs

def get_track2_metric(ans_data, gt_data):
    i_err, s_err, d_err = 0, 0, 0
    num_gt = len(gt_data)
    for ans_ in ans_data:
        pred_ts, pred_cls = ans_[0], ans_[1]
        gt_ts = list(gt_data.keys())
        i_err_flag = True
        
        for gt_ts_ in gt_ts:
            gt_ts_start, gt_ts_end = gt_ts_[0], gt_ts_[1]
            if pred_ts >= gt_ts_start and pred_ts <= gt_ts_end:
                i_err_flag = False
                gt_cls = gt_data[gt_ts_]
                if gt_cls != pred_cls:
                    s_err += 1
                del(gt_data[gt_ts_])
                break
        if i_err_flag:
            i_err += 1
            
    d_err = len(gt_data)
    if num_gt == 0:
        err_score = float(s_err + d_err)
    else:
        err_score = float(i_err + s_err + d_err) / float(num_gt)
    
    return s_err, i_err, d_err, err_score
    
def get_gt_data(gt_idx, gt_dict):
    data = gt_dict[gt_idx]
    gt = {}
    for spk, vad in zip(data['speaker'], data['on_offset']):
        gt[int(vad[0] * fs), int(vad[1] * fs)] = spk
    return gt
    
def main(GT_PATH=None, ANSWER_PATH=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ans', '-a', type = str, required = False, help = 'Predicted answer json file')
    parser.add_argument('--gt',  '-g', type = str, required = False, help = 'Ground truth json file')
    parser.add_argument('--verbose', '-v', type = int, required = False, default = 1, help = 'Verbose level')
    args = parser.parse_args()

    if GT_PATH == None:
        GT_PATH = args.gt
    if ANSWER_PATH == None:
        ANSWER_PATH = args.ans

    with open(ANSWER_PATH, 'r') as f:
        ans_json = json.load(f)
    
    with open(GT_PATH, 'r') as f:
        gt_json = json.load(f)
    
    task2_score = []
    task2_data = ans_json
    for gt_idx in task2_data:
        ans_data = task2_data[gt_idx]
        ans = {}
        for spk, vad in zip(ans_data['speaker'], ans_data['on_offset']):
            ans[(int(vad[0] * fs) + int(vad[1] * fs))//2] = spk
        ans = sorted(ans.items())
        gt = get_gt_data(gt_idx, gt_json)
        
        num_gt = len(gt)
        i_err, s_err, d_err, err_score = get_track2_metric(ans, gt)
        if args.verbose >= 1:
            print('\'{}\' score: ins-{}, sub-{}, del-{}, num-{}, score-{}'.format(gt_idx, i_err, s_err, d_err, num_gt, err_score))
        task2_score = np.append(task2_score, err_score)
    print('Task 2 score: {:.4f}'.format(np.mean(task2_score)))


if __name__ == '__main__':
    main()

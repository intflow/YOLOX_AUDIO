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
    err_score = float(i_err + s_err + d_err) / float(num_gt)
    
    return s_err, i_err, d_err, err_score
    
def get_gt_data(gt_idx, gt_dict):
    data = gt_dict[gt_idx]
    gt = {}
    for spk, vad in zip(data['speaker'], data['on_offset']):
        gt[int(vad[0] * fs), int(vad[1] * fs)] = spk
    return gt

def get_gt_data_intflow(set_idx, drone_idx, gt_dict):
    data = gt_dict[set_idx][drone_idx - 1]['drone_{}'.format(drone_idx)][0]
    gt = {}
    for human in ['M', 'W', 'C']:
        times = data[human]
        if times == 'NONE': break
        for ts in times:
            time_0, time_1 = ts.split('~')[0], ts.split('~')[1]
            time_0, time_1 = dec2sam(time_0), dec2sam(time_1)
            gt[int(time_0), int(time_1)] = human
    gt = sorted(gt.items())
    gt = dict(gt)
    return gt
    
def main(ans, gt):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ans', '-a', type = str, required = True, help = 'Predicted answer json file')
    parser.add_argument('--gt',  '-g', type = str, required = True, help = 'Ground truth json file')
    parser.add_argument('--verbose', '-v', type = int, required = False, default = 0, help = 'Verbose level')
    parser.add_argument('--version', '-ver', type = str, required = False, default = 'intflow', help = 'gist or intflow')
    args = parser.parse_args()

    with open(args.ans, 'r') as f:
        ans_json = json.load(f)
    
    with open(args.gt, 'r') as f:
        gt_json = json.load(f)
    
    task2_score = []
    # extract result from answer json
    if args.version == 'intflow':
        task2_data = ans_json[1]['task2_answer'][0]
        task2_gt_data = gt_json[1]['task2_answer'][0]
        for set_idx in task2_data:
            set_data = task2_data[set_idx]
            gt_set_idx = 'set%02d' %int(set_idx.split('_')[1])
            for idx, drone in enumerate(set_data):
                drone_idx = idx + 1
                drone_data = drone['drone_{}'.format(drone_idx)]
                gt_drone_idx = 'drone%02d' %(drone_idx)
                ans = {}
                for human in ['M', 'W', 'C']:
                    times = drone_data[0][human]
                    for dec_time in times:
                        if dec_time == 'NONE': break
                        sample_time = dec2sam(dec_time)
                        ans[sample_time] = human
                ans = sorted(ans.items())

                gt_idx = set_idx + '_' + 'drone_{}'.format(drone_idx)
                gt = get_gt_data_intflow(set_idx, drone_idx, task2_gt_data)
                
                num_gt = len(gt)
                i_err, s_err, d_err, err_score = get_track2_metric(ans, gt)
                if args.verbose >= 1:
                    print('\'{}\' score: ins-{}, sub-{}, del-{}, num-{}, score-{}'.format(gt_idx, i_err, s_err, d_err, num_gt, err_score))
                task2_score = np.append(task2_score, err_score)
        print('Task 2 score: {:.4f}'.format(np.mean(task2_score)))
        
    elif args.version == 'gist':
        for set_idx in task2_data:
            set_data = task2_data[set_idx]
            gt_set_idx = 'set%02d' %int(set_idx.split('_')[1])
            for idx, drone in enumerate(set_data):
                drone_idx = idx + 1
                drone_data = drone['drone_{}'.format(drone_idx)]
                gt_drone_idx = 'drone%02d' %(drone_idx)
                ans = {}
                for human in ['M', 'W', 'C']:
                    times = drone_data[0][human]
                    for dec_time in times:
                        if dec_time == 'NONE': break
                        sample_time = dec2sam(dec_time)
                        ans[sample_time] = human
                ans = sorted(ans.items())
                gt_idx = gt_set_idx + '_' + gt_drone_idx
                gt = get_gt_data(gt_idx, gt_json)
                
                num_gt = len(gt)
                i_err, s_err, d_err, err_score = get_track2_metric(ans, gt)
                if args.verbose >= 1:
                    print('\'{}\' score: ins-{}, sub-{}, del-{}, num-{}, score-{}'.format(gt_idx, i_err, s_err, d_err, num_gt, err_score))
                task2_score = np.append(task2_score, err_score)
        print('Task 2 score: {:.4f}'.format(np.mean(task2_score)))


if __name__ == '__main__':
    main()

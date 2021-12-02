import json
import sys
from pytictoc import TicToc



def main():

    DATA_PATH = '/data/AIGC_3rd_2021/GIST_tr2_veryhard500/wav'
        
    t = TicToc()
    t.tic()

    import infer_audio
    json_out = infer_audio.run_audio_infer(DATA_PATH)
    

    print("----Write Answer:----")
    with open("output/tr2_devel_500_est.json", "w", encoding='UTF-8') as json_file:
        json.dump(json_out, json_file, indent=2, ensure_ascii=False)

    ## 6. Evaluate scores (Target, Estimates)
    import metric_audio
    metric_audio.main('/data/AIGC_3rd_2021/GIST_tr2_veryhard500/tr2_devel_500.json', 'output/tr2_devel_500_est.json')
    t.toc()

if __name__ == '__main__':
    main()
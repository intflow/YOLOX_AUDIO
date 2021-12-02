import json
import sys
from pytictoc import TicToc



def main():

    DATA_PATH = '/data/AIGC_3rd_2021/GIST_tr2_1000/wav'
    OUTPUT_PATH = "output/tr2_devel_1000_est.json"
    ANSWER_PATH = '/data/AIGC_3rd_2021/GIST_tr2_1000/tr2_devel_1000.json'
        
    t = TicToc()
    t.tic()

    import infer_audio
    json_out = infer_audio.run_audio_infer(DATA_PATH)
    

    print("----Write Answer:----")
    with open(OUTPUT_PATH, "w", encoding='UTF-8') as json_file:
        json.dump(json_out, json_file, indent=2, ensure_ascii=False)

    ## 6. Evaluate scores (Target, Estimates)
    import metric_audio
    metric_audio.load_json(ANSWER_PATH, OUTPUT_PATH)
    t.toc()

if __name__ == '__main__':
    main()
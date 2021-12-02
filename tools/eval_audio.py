import json
import sys
from pytictoc import TicToc



def main():

    DATA_PATH = '/home/agc2021/dataset'
        
    t = TicToc()
    t.tic()

    print("0. Load .json template")
    with open('assets/answersheet_3_00_template.json', encoding='UTF-8') as json_file:
        json_data = json.load(json_file)


    import infer_audio
    json_out = infer_audio.run_audio_infer(DATA_PATH)
    json_data[1] = json_out[0]


    print("----Write Answer:----")
    with open("answersheet_3_00_kmjeon.json", "w", encoding='UTF-8') as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)

    ## 6. Evaluate scores (Target, Estimates)
    if SUBMIT_FLAG == 0:
        import AIGC_answer
        AIGC_answer.load_json('assets/answersheet_3_00_AIGC2021.json', 'answersheet_3_00_kmjeon.json')
        t.toc()

if __name__ == '__main__':
    main()
import os
import json




output_path_ls = ['../results/test_entailment_bank_task2_gpt_4o_mini_SWAP', 
                  '../results/test_entailment_bank_task2_false_hypothesis_gpt_4o_mini_SWAP',
                  '../results/test_entailment_bank_task2_not_enough_evidence_hypothesis_manual_gpt_4o_mini_SWAP']
for output_path in output_path_ls:
    print(output_path)
    cnt = 0
    cnt_correct = 0
    for file in os.listdir(output_path):
        if file.endswith('.json'):
            with open(f'{output_path}/{file}', 'r') as f:
                data = json.load(f)
                if 'flag_correct' in data:
                    print(file)
                    cnt += 1
                    if data['flag_correct']:
                        cnt_correct += 1

    print(cnt_correct, cnt, cnt_correct/cnt)
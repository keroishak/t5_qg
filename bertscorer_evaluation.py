from bert_score import BERTScorer
import torch
import json
import re
import csv

scorer = BERTScorer(model_type= "microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)

candidate_files_path = ['data/test_dataset_v2_BT_t5_small_5_epochs_without_rejection_8_june.csv', 'data/test_dataset_v2_BT_t5_small_5_epochs_seq_rej_3500_debug_1e4lr_12_june.csv', 'data/test_dataset_v2_BT_t5_base_debug_7e5lr_5_epochs_16_june.csv', 'data/test_dataset_v2_BT_t5_base_debug_7e5lr_seq_rejection_3500_5_epochs_16_june.csv']

def evaluate(refs, candidates):
    P, R, F1 = scorer.score(candidates, refs)
    return P, R, F1

def get_candidates_and_references(file_path):
    references = []
    candidates = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            golden_question = row['golden question'].strip()
            generated_qg = row['generated question - top k'].replace('<pad>', '').replace('</s>', '').strip()
            references.append(golden_question)
            candidates.append(generated_qg)
    return references, candidates 

def get_file_content(file_path):
    result = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            line = re.sub(r' +', ' ', line) # remove addition spaces
            result.append(line)
    return result

def convert_into_list(np_array):
    result = []
    for x in np_array:
        result.append(str(x))
    return result

def get_file_content3(file_path):
    with open(file_path, "r") as file:
        refs = []
        generated_p = []
        generated_beam = []
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count==0:
                line_count+=1
                continue
            top_p = row[3]
            generated_p.append(top_p.strip())
            generated_beam.append(row[2].strip())
            ref = row[1]
            refs.append(ref)
            line_count+=1
        
    return refs, generated_beam, generated_p

def run_evaluation():
    # refs = get_file_content('data/'+ reference_file_path)
    # for file_name in candidate_files_path:
    #     candidate = get_file_content('data/'+ file_name)
    for file_name in candidate_files_path:
        references, candidates = get_candidates_and_references(file_name)
        P, R, F1 = evaluate(references, candidates)
        P_list = convert_into_list(P.numpy())
        R_list = convert_into_list(R.numpy())
        F1_list = convert_into_list(F1.numpy())

        evaluation_result = {
            'precisions': P_list,
            'recalls': R_list,
            'f1_scores': F1_list,
            'average_precision': str(torch.mean(P).item()),
            'average_recall': str(torch.mean(R).item()),
            'average_f1': str(torch.mean(F1).item()),
        }
        file_name_split_parts = file_name.split('.')
        output_file = '.'.join(file_name_split_parts[:-1]) + '_bertscorer_evaluation.json'
        output_path = output_file
        with open(output_path, 'w') as outfile:
            json.dump(evaluation_result, outfile)

def run_evaluation_2():
    for i, file_name in enumerate(candidate_files_path):
        print(file_name)
        print()
        refs, generated_beam, generated_p = get_file_content3(file_name)
        beam_P, beam_R, beam_F1 = evaluate(refs, generated_beam)
        # beam_P_list = convert_into_list(beam_P.numpy())
        # beam_R_list = convert_into_list(beam_R.numpy())
        # beam_F1_list = convert_into_list(beam_F1.numpy())

        evaluation_result = {
            'average_precision': str(torch.mean(beam_P).item()),
            'average_recall': str(torch.mean(beam_R).item()),
            'average_f1': str(torch.mean(beam_F1).item()),
        }
        print("beam search")
        print(evaluation_result)
        p_P, p_R, p_F1 = evaluate(refs, generated_p)
        evaluation_result = {
            'average_precision': str(torch.mean(p_P).item()),
            'average_recall': str(torch.mean(p_R).item()),
            'average_f1': str(torch.mean(p_F1).item()),
        }
        print("top-p generation")
        print(evaluation_result)
        print("-----------------------------")

if __name__=='__main__':
    run_evaluation_2()        
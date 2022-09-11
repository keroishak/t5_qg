from bleu import compute_bleu
import re
import nltk
import csv
from nlgeval import NLGEval

reference_file_path = 'references.txt'
candidate_files_path = ['data/test_dataset_v2_BT_t5_base_debug_7e5lr_10_epochs_3500_seq_rej_21_june.csv']
def get_file_content(file_patbh):
    result = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            line = re.sub(r' +', ' ', line) # remove addition spaces
            result.append(line)
    return result

def get_file_content2(file_path):
    with open(file_path, "r") as file:
        refs = []
        generated_q = {"beam1":[], "beam2":[], "beam3":[],"beam4": [], "top_k":[]}
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count==0:
                line_count+=1
                continue
            beam_q = row[3].split('\n')
            beam_q[0] = beam_q[0].replace('<pad>', '').replace('</s>', '').strip()
            beam_q[1] = beam_q[1].replace('<pad>', '').replace('</s>', '').strip()
            beam_q[2] = beam_q[2].replace('<pad>', '').replace('</s>', '').strip()
            beam_q[3] = beam_q[3].replace('<pad>', '').replace('</s>', '').strip()
            top_k_q = row[4].replace('<pad>', '').replace('</s>', '').strip()
            generated_q["beam1"].append(beam_q[0])
            generated_q["beam2"].append(beam_q[1])
            generated_q["beam3"].append(beam_q[2])
            generated_q["beam4"].append(beam_q[3])
            generated_q["top_k"].append(top_k_q)
            ref = row[2]
            refs.append(ref)
            line_count+=1
        
    return refs, generated_q

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

def tokenize(str_list):
    result = []
    for s in str_list:
        tokenized_s = nltk.word_tokenize(s)
        result.append(tokenized_s)
    return result    


def run_evaluation():
    for file_name in candidate_files_path:
        refs, generated_q = get_file_content2(file_name)
        tokenized_refs = tokenize(refs)
        for key, value in generated_q.items():
            tokenized_candidates = tokenize(value)
            bleu, precisions, bp, ratio, question_length, reference_length = compute_bleu(tokenized_refs, tokenized_candidates, max_order=4)
            print("filename: "+ file_name)
            print("key: "+ key)
            print("BLEU: %f" % (bleu))
            print("Precisions: ", precisions)
            print("Brevity Penalty: %f" % (bp))
            print("Ratio: %f" % (ratio))
            print("Question length: %f" % (question_length))
            print("Reference length: %f" % (reference_length))
        print("\n")

def run_nlg_eval():
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
    for i, file_name in enumerate(candidate_files_path):
        refs, generated_beam, generated_p = get_file_content3(file_name)
        metrics_dict1 = nlgeval.compute_metrics([refs], generated_beam)
        print("beam search generation")
        print(metrics_dict1)
        metrics_dict2 = nlgeval.compute_metrics([refs], generated_p)
        print("top-p generation")
        print(metrics_dict2)
        print("-----------------------------")


        

if __name__=='__main__':
    run_nlg_eval()
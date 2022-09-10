# Train command
python train.y
or
python train_loss_truncation.py

# Evaluate on test dataset
#### 1. Run Inference on test dataset
python evaluate.py --data_file data/test_dataset.tfrecord --beam_search 1 --top_k 1
#### 1. Run BLEU evaluation or BERTScore
open the bleu_evaluation.py or bertscorer_evaluation.py and modify the candidate_files_path list with the files that are going to be evaluated, then run the following script.
- python bleu_evaluation.py
- python bertscorer_evaluation.py


The predictions were made using the fine-tuned version of ESMBind and runs it through the WALLE set of 1723 samples, but 145 of them have sequence length greater than 1024 so excluded from the analysis. 

Raw data are located under abag_dataset/processed/fine-tune-input 
- abdbids.txt : the list of sample ids
- labels.pkl  : residue level labels
- seqres.pkl  : antigen sequences
All three files have same sample order 

Prediction results are located at experiments/inference-fine-tuned-esmbind/result.pkl 

mcc scores for the 1578 samples are located at experiments/inference-fine-tuned-esmbind/all_mcc(1578).pkl 

warn sample indices were saved at experiments/inference-fine-tuned-esmbind/warn_indices.pkl
the indices are 0-based and corresponding to the ababids.txt file 

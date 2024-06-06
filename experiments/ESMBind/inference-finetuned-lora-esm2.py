import os
import pickle
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
# Imports specific to the custom peft lora model
from peft import PeftModel
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ==================== Configuration ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE =Path(__file__).resolve().parent
DATA = BASE/"binding_sites_random_split_by_family_550K"
INTERIM = BASE/"interim"

# ==================== Main ====================
if __name__ == "__main__":
    # --------------------
    # Model
    # --------------------
    # Path to the base model and the saved LoRA model
    base_model_path = "facebook/esm2_t12_35M_UR50D"
    base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
    model_path = BASE/"lora_binding_sites"/"best_model_esm2_t12_35M_lora_2023-11-09_12-02-07"
    loaded_model = PeftModel.from_pretrained(base_model, model_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Accelerator prep
    accelerator = Accelerator()
    loaded_model = accelerator.prepare(loaded_model)
    loaded_model.eval()

    # --------------------
    # Data
    # --------------------
    # # Protein sequence for inference
    # protein_sequence = [
    #     "MAVPETRPNHTIYINNLNEKIKKDELKKSLHAIFSRFGQILDILVSRSLKMRGQAFVIFKEVSSATNALRSMQGFPFYDKPMRIQYAKTDSDIIAKMKGT",  # Replace with your actual sequence
    #     "MAVPETRPNHTIYINNLNEKIKKDELKKSLHAIFSRFGQILDILVSRSLKMRGQAFV",  # Replace with your actual sequence
    # ]

    # read antigen sequences from file
    import pickle
    with open(BASE/'abag_dataset'/'processed'/'fine-tune-input'/'seqres.pkl', 'rb') as f:
        protein_sequence = pickle.load(f)

    # --------------------
    # batch inference
    # --------------------
    # Process in Batches with Accelerator
    batch_size = 32  # Adjust based on GPU memory
    num_batches = len(protein_sequence) // batch_size + int(len(protein_sequence) % batch_size != 0)

    all_logits = []
    all_input_ids = []
    for i in tqdm(range(num_batches)):
        batch_seq = protein_sequence[i * batch_size:(i + 1) * batch_size]
        inputs = loaded_tokenizer(batch_seq, return_tensors="pt",
                                  truncation=True, max_length=1024,
                                  padding='max_length')
        # Move inputs to the same device as the model
        all_input_ids.append(inputs["input_ids"])
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        # Use Accelerator to manage computations
        with accelerator.no_sync(loaded_model):  # During inference (model evaluation), gradient computation and synchronization are unnecessary. Wrapping the inference code with no_sync ensures that the model doesn't perform these redundant operations, leading to more efficient execution.
            inputs = accelerator.prepare(inputs)
            with torch.no_grad():
                batch_logits = loaded_model(**inputs).logits
                all_logits.append(accelerator.gather(batch_logits))  # Gather outputs from all processes, essential in multi-GPU setups

    # Concatenate logits from all batches
    all_logits = torch.cat(all_logits, dim=0)  # torch.Size([1723, 1024, 2])
    all_input_ids = torch.cat(all_input_ids, dim=0)  # torch.Size([1723, 1024])

    # convert logits to predictions
    predictions = torch.argmax(all_logits, dim=2)  # torch.Size([2, 1024])

    # convert logits to probabilities
    probs = torch.nn.functional.softmax(all_logits, dim=2)  # torch.Size([2, 1024, 2])

    results = []
    for i, raw_seq in tqdm(enumerate(protein_sequence), total=len(protein_sequence)):
        tokens = loaded_tokenizer.convert_ids_to_tokens(all_input_ids[i])  # Convert input ids back to tokens
        idx = [j for j, t in enumerate(tokens) if t not in ['<pad>', '<cls>', '<eos>']]  # exclude special tokens
        results.append({
            'raw_seq': raw_seq,  # str
            'preds'  : predictions[i, idx].cpu().numpy(),   # binary
            'probs'  : probs[i, idx, :].cpu().numpy(),      # probabilities softmax
            'logits' : all_logits[i, idx, :].cpu().numpy()  # logits
        })

    # save results to local
    with open(BASE/'experiments'/'inference-fine-tuned-esmbind'/'result.pkl', 'wb') as f:
        pickle.dump(results, f)

    # TODO: Calculate metrics

    # 0. load results
    with open(BASE/'experiments'/'inference-fine-tuned-esmbind'/'result.pkl', 'rb') as f:
        results = pickle.load(f)

    # 1. load labels
    with open(BASE/'abag_dataset'/'processed'/'fine-tune-input'/'labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    # 2. calculate mcc metrics
    all_mcc = []
    warn_items = 0
    warn_indices = []
    for i, (x, y) in enumerate(zip(labels, results)):
        if len(x) == len(y['preds']):
            all_mcc.append(matthews_corrcoef(x, y['preds']))
        else:
            # warning that a sample may have exceeded the max length the model can accept
            print(f"Warning: a sample has length {len(x)} that exceeds the max length 1024 the model can accept:\n{y['raw_seq']}\n")
            warn_items += 1
            warn_indices.append(i)
    print(warn_items)  # 145

    # save all mcc to local
    with open(BASE/'experiments'/'inference-fine-tuned-esmbind'/'all_mcc(1578).pkl', 'wb') as f:
        pickle.dump(all_mcc, f)
    # read all_mcc
    with open(BASE/'experiments'/'inference-fine-tuned-esmbind'/'all_mcc(1578).pkl', 'rb') as f:
        all_mcc = pickle.load(f)

    # save warn indices to local
    with open(BASE/'experiments'/'inference-fine-tuned-esmbind'/'warn_indices.pkl', 'wb') as f:
        pickle.dump(warn_indices, f)

    # 3. draw metrics distribution
    import matplotlib.pyplot as plt
    plt.hist(all_mcc, bins=50)
    plt.title('MCC Distribution')
    plt.xlabel('MCC')
    plt.ylabel('Count')
    plt.savefig(BASE/'experiments'/'inference-fine-tuned-esmbind'/'mcc_histogram.pdf')
    plt.show()

    # 3. draw metrics distribution as violin
    import seaborn as sns
    sns.violinplot(all_mcc)
    plt.title('MCC Distribution')
    plt.xlabel('MCC')
    plt.savefig(BASE/'experiments'/'inference-fine-tuned-esmbind'/'mcc_violin.pdf')
    plt.show()

    # get the quartiles
    q1, q3 = np.percentile(all_mcc, [25, 75])
    median = np.median(all_mcc)
    q1, median, q3
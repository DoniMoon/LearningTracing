import json
import os
import random
import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

llm_list = [
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen1.5-1.8B-Chat",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-1b-it",
    "google/gemma-2b-it",
    "google/gemma-2b",
    "google/gemma-3-270m-it",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-1.1-2b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/phi-2",
    "jinaai/ReaderLM-v2",
    "facebook/MobileLLM-R1-950M",
]

DATASETS = ["507_statics", "1148_biology", "eedi_2020"]
BASE_DIR = "Datas"
BATCH_SIZE = 2
CHOICE_TOKENS = ["1", "2", "3", "4"]

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def get_target_token_ids(tokenizer):
    ids = []
    for t in CHOICE_TOKENS:
        token_id = tokenizer.encode(t, add_special_tokens=False)[-1]
        ids.append(token_id)
    return ids

def sync_datasets(dset_name):
    source_path = os.path.join(BASE_DIR, dset_name, "mcqs.json")
    target_path = os.path.join(BASE_DIR, dset_name, "mcqs_result.json")
    
    source_data = load_json(source_path)
    target_data = load_json(target_path)
    
    if not target_data:
        target_data = copy.deepcopy(source_data)
    else:
        for qid, kcs in source_data.items():
            if qid not in target_data:
                target_data[qid] = copy.deepcopy(kcs)
            else:
                for kc_idx, kc_obj in kcs.items():
                    if kc_idx not in target_data[qid]:
                        target_data[qid][kc_idx] = copy.deepcopy(kc_obj)
    
    return target_data, target_path

def format_prompt_content(question_data, shuffled_indices):
    q_text = question_data['question']
    answer = question_data['answer']
    distractors = question_data['distractors']
    
    options_pool = [answer] + distractors
    final_options = [options_pool[i] for i in shuffled_indices]
    
    option_text = ""
    for idx, opt in enumerate(final_options):
        option_text += f"{idx+1}. {opt}\n"
    
    content = f"""{q_text}

{option_text}
Select the correct answer from options 1, 2, 3, or 4. Return only the number corresponding to the correct option.
Answer:"""
    return content

def apply_template(tokenizer, content):
    messages = [{"role": "user", "content": content}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return f"User: {content}\nAssistant:"

def main():
    for model_name in llm_list:
        print(model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left' 
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                torch_dtype="auto", 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        model.eval()
        choice_ids = get_target_token_ids(tokenizer)
        
        dataset_queues = {}
        dataset_paths = {}
        full_data = {}
        
        for dset in DATASETS:
            data, path = sync_datasets(dset)
            full_data[dset] = data
            dataset_paths[dset] = path
            
            tasks = []
            for qid, kcs in data.items():
                for kc_idx, kc_obj in kcs.items():
                    if 'solved' not in kc_obj:
                        kc_obj['solved'] = {}
                    if model_name not in kc_obj['solved']:
                        tasks.append((qid, kc_idx))
            
            dataset_queues[dset] = tasks
            
        while any(len(q) > 0 for q in dataset_queues.values()):
            for dset in DATASETS:
                queue = dataset_queues[dset]
                if not queue:
                    continue
                
                batch_tasks = queue[:BATCH_SIZE]
                dataset_queues[dset] = queue[BATCH_SIZE:]
                
                if not batch_tasks:
                    continue

                batch_prompts = []
                batch_meta = []
                
                for qid, kc_idx in batch_tasks:
                    kc_obj = full_data[dset][qid][kc_idx]
                    indices = [0, 1, 2, 3]
                    random.shuffle(indices)
                    
                    content = format_prompt_content(kc_obj, indices)
                    prompt = apply_template(tokenizer, content)
                    
                    batch_prompts.append(prompt)
                    batch_meta.append({
                        "qid": qid,
                        "kc_idx": kc_idx,
                        "indices": indices
                    })
                
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Get logits of the last token
                    next_token_logits = outputs.logits[:, -1, :]
                    target_logits = next_token_logits[:, choice_ids]
                
                target_logits_list = target_logits.tolist()
                
                for i, meta in enumerate(batch_meta):
                    qid = meta['qid']
                    kc_idx = meta['kc_idx']
                    indices = meta['indices']
                    logits_1234 = target_logits_list[i]
                    
                    reordered_logits = [0.0] * 4
                    for rank, original_idx in enumerate(indices):
                        reordered_logits[original_idx] = logits_1234[rank]
                    
                    probs = F.softmax(torch.tensor(reordered_logits), dim=0).tolist()
                    
                    # Argmax over the 1,2,3,4 logits
                    pred_token_idx = torch.tensor(logits_1234).argmax().item()
                    pred_original_idx = indices[pred_token_idx]
                    
                    is_correct = 1 if pred_original_idx == 0 else 0
                    
                    full_data[dset][qid][kc_idx]['solved'][model_name] = {
                        "logits": reordered_logits,
                        "is_correct": is_correct,
                        "correct_probability": probs[0]
                    }
                
                save_json(dataset_paths[dset], full_data[dset])
                print(f"[{model_name}] Processed {len(batch_tasks)} items for {dset}")

        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

# TRAINING ORDER (sequential):
#   C-STANCE  FOMC  MeetingBank  Py150  ScienceQA  NumGLUE-cm
#
# USAGE:
#   python train_6tasks_PMN_PSN.py \
#     --data_path data/LLM-CL_Benchmark/... \
#     --model_name_or_path models/llama-2-7b-chat \
#     --output_dir outputs/6tasks_v1_$(date +%Y%m%d_%H%M) \
#     --epochs_CSTANCE 1 --epochs_FOMC 1 --epochs_MeetingBank 1 \
#     --epochs_Py150 1 --epochs_ScienceQA 1 --epochs_NumGLUE_cm 1 \
#     --lr_CSTANCE_FOMC 1e-6 --lr_MeetingBank 3e-5 \
#     --lr_Py150 5e-5 --lr_ScienceQA 1e-6 --lr_NumGLUE_cm 5e-5 \
#     --bf16

import os
import json
import re
import torch
import argparse
import torch.distributed as dist
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments, default_data_collator, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


def is_main_process():
    """Check if current process is the main (rank 0) process."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def clean_scienceqa_answer(answer: str) -> str:
    answer = str(answer).strip()
    if not answer:
        return "a"
    first_char = answer[0].upper()
    if first_char in {"A", "B", "C", "D"}:
        return first_char.lower()
    for c in answer:
        if c.upper() in {"A", "B", "C", "D"}:
            return c.lower()
    return "a"


def tokenize_choice(examples, tokenizer, choice_tokens, max_length=1024):
    input_ids = []
    labels = []
    valid_choices = set(choice_tokens.keys())
    
    for prompt, ans in zip(examples["prompt"], examples["answer"]):
        gold = str(ans).strip().lower()
        if gold.startswith("(") and len(gold) >= 2:
            gold = gold[1]
        elif gold in {"strongly oppose", "oppose", "support", "strongly support"}:
            mapping = {"strongly oppose": "a", "oppose": "b", "support": "c", "strongly support": "d"}
            gold = mapping[gold]
        else:
            gold = next((c for c in gold if c in valid_choices), "a")
        
        if gold not in valid_choices:
            continue

        prompt_clean = " ".join(str(prompt).split()) + " "
        prompt_tokens = tokenizer(prompt_clean, add_special_tokens=True, truncation=False)["input_ids"]
        if len(prompt_tokens) > max_length - 1:
            prompt_tokens = prompt_tokens[-(max_length - 1):]

        answer_token_id = choice_tokens[gold]
        full_input = prompt_tokens + [answer_token_id]
        full_labels = [-100] * len(prompt_tokens) + [answer_token_id]

        if len(full_input) < max_length:
            pad_len = max_length - len(full_input)
            full_input += [tokenizer.pad_token_id] * pad_len
            full_labels += [-100] * pad_len
        else:
            full_input = full_input[:max_length]
            full_labels = full_labels[:max_length]

        input_ids.append(full_input)
        labels.append(full_labels)

    return {"input_ids": input_ids, "labels": labels}



def format_instruction_meetingbank(prompt, answer):
    return f"""<s>[INST] <<SYS>>
You are a helpful assistant. Summarize the following meeting transcript in one concise paragraph.
<</SYS>>

{prompt} [/INST] {answer}</s>"""


def tokenize_generation_meetingbank(examples, tokenizer, max_length=512):
    prompts = [str(p).strip() for p in examples["prompt"]]
    answers = [str(a).strip() for a in examples["answer"]]
    texts = [format_instruction_meetingbank(p, a) for p, a in zip(prompts, answers)]
    tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    input_ids = []
    for ids in tokenized["input_ids"]:
        if len(ids) < max_length:
            ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        input_ids.append(ids)
    return {"input_ids": input_ids, "labels": input_ids}



def tokenize_py150(examples, tokenizer, max_length=1024, target_token_len=8):
    input_ids = []
    labels = []
    bos_token_id = tokenizer.bos_token_id
    eol_token = "<EOL>"
    
    def score_line(line: str) -> int:
        s = line.strip()
        if not s:
            return 0
        if s.startswith(("def ", "class ")):
            return 20
        if s.endswith(("(", "[", "{")):
            return 9
        if "=" in s and not any(op in s for op in ("==", "!=", ">=", "<=", "+=", "-=")) and s[0].isalpha():
            return 8
        if s.endswith(":"):
            return 6
        if s.startswith(("if ", "for ", "while ", "try:", "except")):
            return 5
        return 1

    for prompt, answer in zip(examples["prompt"], examples["answer"]):
        prompt_str = str(prompt)
        answer_str = str(answer).rstrip()

        answer_ids = tokenizer(answer_str, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            continue
        answer_ids = answer_ids[:target_token_len]

        lines = [line + eol_token for line in prompt_str.split(eol_token) if line.strip()]
        if not lines:
            continue

        line_info = []
        for line in lines:
            tok_ids = tokenizer(line, add_special_tokens=False)["input_ids"]
            if tok_ids:
                line_info.append({
                    'text': line,
                    'ids': tok_ids,
                    'score': score_line(line)
                })

        if not line_info:
            continue

        max_prompt_tokens = max_length - 1 - len(answer_ids)
        if max_prompt_tokens <= 0:
            continue

        selected_ids = []
        current_len = 0
        n_lines = len(line_info)

        for i in range(n_lines - 1, -1, -1):
            line_data = line_info[i]
            boost = 3 if (n_lines - 1 - i) < 3 else 0
            effective_score = line_data['score'] + boost

            if current_len + len(line_data['ids']) <= max_prompt_tokens:
                selected_ids = line_data['ids'] + selected_ids
                current_len += len(line_data['ids'])

        recent_def_class = None
        for line_data in reversed(line_info):
            if line_data['score'] >= 20:
                recent_def_class = line_data
                break

        if recent_def_class is not None:
            found_in_recent = any(
                item['text'] == recent_def_class['text']
                for item in line_info[-10:]
            )
            needed = len(recent_def_class['ids'])
            if not found_in_recent and current_len + needed <= max_prompt_tokens:
                selected_ids = recent_def_class['ids'] + selected_ids
                current_len += needed

        if not selected_ids:
            all_prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            if len(all_prompt_ids) > max_prompt_tokens:
                selected_ids = all_prompt_ids[-max_prompt_tokens:]
            else:
                selected_ids = all_prompt_ids

        if not selected_ids or selected_ids[0] != bos_token_id:
            if len(selected_ids) >= max_length - len(answer_ids):
                selected_ids = selected_ids[-(max_length - len(answer_ids) - 1):]
            selected_ids = [bos_token_id] + selected_ids

        if len(selected_ids) + len(answer_ids) > max_length:
            max_answer = max_length - len(selected_ids)
            if max_answer <= 0:
                continue
            answer_ids = answer_ids[:max_answer]

        full_ids = selected_ids + answer_ids
        full_labels = [-100] * len(selected_ids) + answer_ids
        input_ids.append(full_ids)
        labels.append(full_labels)

    return {"input_ids": input_ids, "labels": labels}


def clean_numglue_prompt(raw_prompt: str) -> str:
    prompt = raw_prompt.strip()
    prompt = re.sub(r'^Solve the following math problem\.\s*', '', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\s*Question:\s*', '', prompt)
    prompt = re.sub(r'\s*Answer:\s*$', '', prompt)
    names = ["addison", "genesis", "alice", "bob", "charlie", "diana", "john", "mary", 
             "nicholas", "mason", "the student", "the teacher", "biology teacher"]
    for name in names:
        prompt = re.sub(rf"\b{name}\b", "someone", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    if not prompt.endswith('?'):
        prompt = re.sub(r'[.!?]*$', '?', prompt)
    return prompt


def build_numglue_prompt(question: str) -> str:
    cleaned_q = clean_numglue_prompt(question)
    return (
        "<s>[INST] <<SYS>>\n"
        "Solve the math problem step by step. Show your reasoning clearly. "
        "End your response with the final answer as a single number.\n"
        "Example:\n"
        "Alice has 3 apples and gets 2 more. How many?\n"
        "Alice starts with 3 apples. She gets 2 more, so 3 + 2 = 5. The answer is 5.\n"
        "<</SYS>>\n\n"
        f"{cleaned_q} [/INST]"
    )


def clean_numglue_answer(ans):
    try:
        val = float(ans)
        if val != val:
            raise ValueError("NaN")
        if abs(val) < 1e-10:
            return "0"
        if val.is_integer():
            return str(int(val))
        else:
            rounded = round(val, 5)
            s = f"{rounded:.5f}".rstrip('0').rstrip('.')
            return s if s else "0"
    except (ValueError, TypeError):
        numbers = re.findall(r'-?\d+\.?\d*', str(ans))
        return numbers[-1] if numbers else "0"


def tokenize_generation_numglue(examples, tokenizer, max_length=512):
    prompts = [str(p).strip() for p in examples["prompt"]]
    answers = [str(a).strip() for a in examples["answer"]]
    
    input_ids = []
    labels = []

    inst_end_str = "[/INST]"
    inst_end_ids = tokenizer(inst_end_str, add_special_tokens=False)["input_ids"]

    debug_printed = False

    for p, a in zip(prompts, answers):
        if not re.search(r'\d', str(a)):
            continue
            
        base_prompt = build_numglue_prompt(p)
        ans_clean = clean_numglue_answer(a)
        full_text = base_prompt + ans_clean + "</s>"

        ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )["input_ids"]

        answer_start = -1
        for i in range(len(ids) - len(inst_end_ids) + 1):
            if ids[i:i + len(inst_end_ids)] == inst_end_ids:
                answer_start = i + len(inst_end_ids)

        if answer_start == -1:
            base_ids = tokenizer(base_prompt, add_special_tokens=False)["input_ids"]
            answer_start = len(base_ids)

        input_ids.append(ids)
        label_seq = [-100] * answer_start + ids[answer_start:]
        labels.append(label_seq)

        if not debug_printed:
            print("\n" + "="*60)
            print("[DEBUG] First training example inspection:")
            print(f"Base prompt: {repr(base_prompt)}")
            print(f"Full text: {repr(full_text)}")
            print(f"[/INST] token IDs: {inst_end_ids}")
            print(f"Total tokens: {len(ids)}")
            print(f"Answer start index: {answer_start}")
            print(f"First 5 supervised tokens: {[tokenizer.decode([t]) for t in ids[answer_start:answer_start+5]]}")
            print("="*60 + "\n")
            debug_printed = True

    return {"input_ids": input_ids, "labels": labels}


def extract_last_number(text: str) -> str:
    candidates = re.findall(r'-?\d+\.?\d*', text)
    return candidates[-1] if candidates else ""



try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

def compute_rouge_l(predictions, references):
    if not HAS_ROUGE or len(predictions) == 0:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []
    for pred, ref in zip(predictions, references):
        pred = str(pred).strip()
        ref = str(ref).strip()
        if not pred or not ref:
            scores.append(0.0)
            continue
        try:
            score = scorer.score(ref, pred)['rougeL'].fmeasure
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_meetingbank_rouge(model, tokenizer, raw_eval_dataset, device, max_new_tokens=300):
    model.eval()
    predictions = []
    references = []

    for example in raw_eval_dataset:
        prompt = example["prompt"]
        reference = example["answer"]
        references.append(reference)

        inputs = tokenizer(
            f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Summarize the following meeting transcript in one concise paragraph.\n<</SYS>>\n\n{prompt} [/INST]",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
            add_special_tokens=False
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(pred)

    return compute_rouge_l(predictions, references)


@torch.no_grad()
def evaluate_choice_accuracy(model, tokenizer, raw_eval_dataset, device, max_new_tokens=32):
    model.eval()
    correct = 0
    total = 0
    valid_choices = {"a", "b", "c", "d"}

    for example in raw_eval_dataset:
        prompt = example["prompt"]
        gold_choice = example["answer"].strip().lower()
        if gold_choice not in valid_choices:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

        pred_choice = None
        for char in generated:
            if char in valid_choices:
                pred_choice = char
                break
        if pred_choice is None:
            for c in valid_choices:
                if c in generated:
                    pred_choice = c
                    break

        if pred_choice == gold_choice:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_py150_exact_match(model, tokenizer, raw_eval_dataset, device, max_new_tokens=16):
    model.eval()
    correct = 0
    total = 0

    eol_token_id = None
    if "<EOL>" in tokenizer.get_vocab():
        eol_token_id = tokenizer.convert_tokens_to_ids("<EOL>")

    for i, example in enumerate(raw_eval_dataset):
        prompt = example["prompt"]
        gold = example["answer"].strip()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
            padding=False,
            add_special_tokens=False
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            eos_token_id=eol_token_id
        )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

        if "<EOL>" in pred_text:
            pred_text = pred_text.split("<EOL>", 1)[0].strip()

        gold_ids = tokenizer(gold, add_special_tokens=False)["input_ids"]
        pred_ids = tokenizer(pred_text, add_special_tokens=False)["input_ids"]
        min_len = min(len(gold_ids), len(pred_ids), 8)
        match = (gold_ids[:min_len] == pred_ids[:min_len]) if min_len > 0 else (len(gold_ids) == len(pred_ids) == 0)

        if match:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_numglue_exact_match(model, tokenizer, raw_eval_dataset, device, max_new_tokens=512):
    model.eval()
    correct = 0
    total = 0

    for idx, example in enumerate(raw_eval_dataset):
        gold_raw = example["answer"]
        if not re.search(r'\d', str(gold_raw)):
            continue
        
        gold = clean_numglue_answer(gold_raw)

        prompt_text = build_numglue_prompt(example['prompt'])
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
            add_special_tokens=False
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False,
            num_beams=1
        )
        pred_raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        pred_extracted = extract_last_number(pred_raw)
        if not pred_extracted:
            pred_extracted = "INVALID"

        match = False
        if pred_extracted != "INVALID":
            try:
                pred_norm = clean_numglue_answer(pred_extracted)
                match = (pred_norm == gold)
            except:
                match = False

        if idx < 3:
            print(f"[DEBUG] Full Prompt Used: {repr(prompt_text)}")
            print(f"[DEBUG] Gold: '{gold_raw}' → Norm: '{gold}'")
            print(f"[DEBUG] Pred Raw: '{pred_raw}'")
            print(f"[DEBUG] Extracted Number: '{pred_extracted}'")
            print(f"[DEBUG] Match: {match}\n")

        if match:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Epochs per task
    parser.add_argument("--epochs_CSTANCE", type=int, default=0)
    parser.add_argument("--epochs_FOMC", type=int, default=0)
    parser.add_argument("--epochs_MeetingBank", type=int, default=0)
    parser.add_argument("--epochs_Py150", type=int, default=0)
    parser.add_argument("--epochs_ScienceQA", type=int, default=0)
    parser.add_argument("--epochs_NumGLUE_cm", type=int, default=0)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    
    # Training
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--bf16", action="store_true")
    
    # Learning rates
    parser.add_argument("--lr_CSTANCE_FOMC", type=float, default=1e-6)
    parser.add_argument("--lr_MeetingBank", type=float, default=3e-5)
    parser.add_argument("--lr_Py150", type=float, default=5e-5)
    parser.add_argument("--lr_ScienceQA", type=float, default=1e-6)
    parser.add_argument("--lr_NumGLUE_cm", type=float, default=5e-5)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add special tokens for Py150
    special_tokens = ["<EOL>"]
    added = tokenizer.add_tokens(special_tokens, special_tokens=False)
    if added > 0 and is_main_process():
        print(f"Added {added} new tokens: {special_tokens}")

    # Choice token mapping
    valid_choices = {"a", "b", "c", "d"}
    test_str = " a b c d"
    tokens = tokenizer.encode(test_str, add_special_tokens=False)
    if len(tokens) == 4:
        CHOICE_TOKENS = {"a": tokens[0], "b": tokens[1], "c": tokens[2], "d": tokens[3]}
    else:
        CHOICE_TOKENS = {}
        for choice in valid_choices:
            ids = tokenizer.encode(" " + choice, add_special_tokens=False)
            CHOICE_TOKENS[choice] = ids[-1] if ids else tokenizer.convert_tokens_to_ids(choice)

    # Model loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_disable()
    model.print_trainable_parameters()

    # Load all datasets
    tasks = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm"]
    datasets = {}
    eval_datasets = {}
    for task in tasks:
        safe_task = task.replace("-", "_")  
        datasets[task] = load_dataset("json", data_files=os.path.join(args.data_path, task, "train.json"), split="train", cache_dir="/tmp")
        eval_datasets[task] = load_dataset("json", data_files=os.path.join(args.data_path, task, "eval.json"), split="train", cache_dir="/tmp")

    # Clean ScienceQA answers
    datasets["ScienceQA"] = datasets["ScienceQA"].map(lambda ex: {"answer": clean_scienceqa_answer(ex["answer"])})
    eval_datasets["ScienceQA"] = eval_datasets["ScienceQA"].map(lambda ex: {"answer": clean_scienceqa_answer(ex["answer"])})

    # Tokenize all
    tokenized_datasets = {}
    tokenized_datasets["C-STANCE"] = datasets["C-STANCE"].map(
        lambda x: tokenize_choice(x, tokenizer, CHOICE_TOKENS, max_length=args.max_length),
        batched=True, remove_columns=datasets["C-STANCE"].column_names
    )
    tokenized_datasets["FOMC"] = datasets["FOMC"].map(
        lambda x: tokenize_choice(x, tokenizer, CHOICE_TOKENS, max_length=args.max_length),
        batched=True, remove_columns=datasets["FOMC"].column_names
    )
    tokenized_datasets["MeetingBank"] = datasets["MeetingBank"].map(
        lambda x: tokenize_generation_meetingbank(x, tokenizer, max_length=args.max_length // 2),
        batched=True, remove_columns=datasets["MeetingBank"].column_names
    )
    tokenized_datasets["Py150"] = datasets["Py150"].map(
        lambda x: tokenize_py150(x, tokenizer, max_length=args.max_length),
        batched=True, remove_columns=datasets["Py150"].column_names
    )
    tokenized_datasets["ScienceQA"] = datasets["ScienceQA"].map(
        lambda x: tokenize_choice(x, tokenizer, CHOICE_TOKENS, max_length=args.max_length),
        batched=True, remove_columns=datasets["ScienceQA"].column_names
    )
    tokenized_datasets["NumGLUE-cm"] = datasets["NumGLUE-cm"].map(
        lambda x: tokenize_generation_numglue(x, tokenizer, max_length=args.max_length // 2),
        batched=True, remove_columns=datasets["NumGLUE-cm"].column_names
    )

    # Print training plan
    tasks_to_train = []
    if args.epochs_CSTANCE > 0: tasks_to_train.append(f"C-STANCE({args.epochs_CSTANCE})")
    if args.epochs_FOMC > 0: tasks_to_train.append(f"FOMC({args.epochs_FOMC})")
    if args.epochs_MeetingBank > 0: tasks_to_train.append(f"MeetingBank({args.epochs_MeetingBank})")
    if args.epochs_Py150 > 0: tasks_to_train.append(f"Py150({args.epochs_Py150})")
    if args.epochs_ScienceQA > 0: tasks_to_train.append(f"ScienceQA({args.epochs_ScienceQA})")
    if args.epochs_NumGLUE_cm > 0: tasks_to_train.append(f"NumGLUE-cm({args.epochs_NumGLUE_cm})")
    print(f"Will train: {' | '.join(tasks_to_train) if tasks_to_train else 'NO TASKS!'}\n")

    round_results = []

    def evaluate_all():
        if not is_main_process():
            return

        acc_cstance = evaluate_choice_accuracy(model, tokenizer, eval_datasets["C-STANCE"], model.device)
        acc_fomc = evaluate_choice_accuracy(model, tokenizer, eval_datasets["FOMC"], model.device)
        rouge_meeting = evaluate_meetingbank_rouge(model, tokenizer, eval_datasets["MeetingBank"], model.device)
        em_py150 = evaluate_py150_exact_match(model, tokenizer, eval_datasets["Py150"], model.device)
        acc_scienceqa = evaluate_choice_accuracy(model, tokenizer, eval_datasets["ScienceQA"], model.device)
        acc_numglue = evaluate_numglue_exact_match(model, tokenizer, eval_datasets["NumGLUE-cm"], model.device)

        result = {
            "C-STANCE": acc_cstance,
            "FOMC": acc_fomc,
            "MeetingBank": rouge_meeting,
            "Py150": em_py150,
            "ScienceQA": acc_scienceqa,
            "NumGLUE-cm": acc_numglue
        }
        round_results.append(result)
        with open(os.path.join(args.output_dir, "rounds.json"), "w") as f:
            json.dump(round_results, f, indent=2)
        print(f"Round {len(round_results)}: "
              f"C-STANCE={acc_cstance:.3f}, FOMC={acc_fomc:.3f}, MeetingBank={rouge_meeting:.3f}, "
              f"Py150={em_py150:.3f}, ScienceQA={acc_scienceqa:.3f}, NumGLUE-cm={acc_numglue:.3f}")

    # Helper: get training args for choice tasks
    def get_choice_args(lr):
        return TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=args.bf16,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            num_train_epochs=1,
            dataloader_drop_last=False,
            overwrite_output_dir=True,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
        )


    # 1. C-STANCE
    if args.epochs_CSTANCE > 0:
        trainer_args = get_choice_args(args.lr_CSTANCE_FOMC)
        for _ in range(args.epochs_CSTANCE):
            trainer = Trainer(
                model=model,
                args=trainer_args,
                train_dataset=tokenized_datasets["C-STANCE"],
                data_collator=default_data_collator,
            )
            trainer.train()
            evaluate_all()

    # 2. FOMC
    if args.epochs_FOMC > 0:
        trainer_args = get_choice_args(args.lr_CSTANCE_FOMC)
        for _ in range(args.epochs_FOMC):
            trainer = Trainer(
                model=model,
                args=trainer_args,
                train_dataset=tokenized_datasets["FOMC"],
                data_collator=default_data_collator,
            )
            trainer.train()
            evaluate_all()

    # 3. MeetingBank
    if args.epochs_MeetingBank > 0:
        mb_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr_MeetingBank,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            bf16=args.bf16,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            num_train_epochs=1,
            dataloader_drop_last=False,
            overwrite_output_dir=True,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=False,
            optim="paged_adamw_8bit",
            dataloader_num_workers=0,
        )
        for _ in range(args.epochs_MeetingBank):
            trainer = Trainer(
                model=model,
                args=mb_args,
                train_dataset=tokenized_datasets["MeetingBank"],
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            )
            trainer.train()
            evaluate_all()

    # 4. Py150
    if args.epochs_Py150 > 0:
        py150_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr_Py150,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            bf16=args.bf16,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            num_train_epochs=1,
            dataloader_drop_last=False,
            overwrite_output_dir=True,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=False,
            optim="paged_adamw_8bit",
            dataloader_num_workers=0,
        )
        for _ in range(args.epochs_Py150):
            trainer = Trainer(
                model=model,
                args=py150_args,
                train_dataset=tokenized_datasets["Py150"],
                data_collator=default_data_collator,
            )
            trainer.train()
            evaluate_all()

    # 5. ScienceQA
    if args.epochs_ScienceQA > 0:
        trainer_args = get_choice_args(args.lr_ScienceQA)
        for _ in range(args.epochs_ScienceQA):
            trainer = Trainer(
                model=model,
                args=trainer_args,
                train_dataset=tokenized_datasets["ScienceQA"],
                data_collator=default_data_collator,
            )
            trainer.train()
            evaluate_all()

    # 6. NumGLUE-cm
    if args.epochs_NumGLUE_cm > 0:
        numglue_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr_NumGLUE_cm,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            bf16=args.bf16,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            num_train_epochs=1,
            dataloader_drop_last=False,
            overwrite_output_dir=True,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=False,
            optim="paged_adamw_8bit",
            dataloader_num_workers=0,
        )
        for _ in range(args.epochs_NumGLUE_cm):
            trainer = Trainer(
                model=model,
                args=numglue_args,
                train_dataset=tokenized_datasets["NumGLUE-cm"],
                data_collator=default_data_collator,
            )
            trainer.train()
            evaluate_all()

    # Save final model
    if is_main_process():
        model.save_pretrained(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    print("6-TASK Multi-Policy Training Completed!")


if __name__ == "__main__":
    main()

import torch
import argparse
import random
from numpy import random
from tqdm import tqdm
import transformers


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    parser.add_argument('--test_k_tokens', type=int, default=2, help='test length')
    
    args = parser.parse_args()
    return args


def generate_prompt_landmark(tokenizer, n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 100000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    
    
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix, # generate: 1k token
        final_question,
    ]
    
    position = []
    current_pos = 0
    block = 0

    for line in lines:
        input_ids = tokenizer(line, return_tensors="pt").input_ids
        line_length = input_ids.size(1)  # Get the length of the sequence
        position.append((current_pos, current_pos + line_length - 1))  # Store the start and end positions
        if line.startswith("The pass key is"):
            block = current_pos // 256 
        current_pos += line_length  # Update the current position
    
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key), position[2], block


def passkey_retrieval_test(model, tokenizer, device, use_cache=False, n_garbage=60000, seed=666, sequence=None):
    prompt, answer, position, block = generate_prompt_landmark(tokenizer, n_garbage, seed+n_garbage) # 修改 seed 为 seed+n，防止重复
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=True
    )
    
    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    
    is_correct = (model_answer == answer_ids[0]).all().item()
    is_split = is_number_in_range(sequence, position)
    
        
    print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct, is_split, len_token, position


def generate_sequence():
    sequence = [0]  
    increment = 256  

    for i in range(1, 201):  
        next_number = sequence[-1] + increment
        sequence.append(next_number)

    return sequence

def is_number_in_range(sequence, position):
    for num in sequence:
        if position[0] <= num and num <= position[1]:
            return True
    return False

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)

    
    from modeling_longheads import LlamaForCausalLM, LlamaConfig
    # Set RoPE scaling factor
    config = LlamaConfig.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
        )

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
    )
    model = model.to('cuda:0')
    
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size,
        padding_side="right",
        use_fast=False,
    )


    sequence = generate_sequence()

    
    # This is a rough ratio to control the number of texts and tokens
    n_garbage = int(3.75 * args.test_k_tokens * 1024 // 1024 * 1024)
    passed_tests = 0
    total_tokens = 0
    for j in tqdm(range(args.num_tests)):
        is_correct, is_split, len_tokens, position = passkey_retrieval_test(model, tokenizer, device, use_cache=True, n_garbage=n_garbage, seed=j, sequence=sequence)
        
        passed_tests += is_correct
        total_tokens += len_tokens
        if is_correct:
            print(f" Success: {position},\tis_split: {is_split}", end="", flush=True)
        else:
            print(f" [Fails]: {position},\tis_split: {is_split}", end="", flush=True)
    avg_tokens = total_tokens//args.num_tests
    accuracy = float(passed_tests)/args.num_tests
    print("Accuracy on the token length %d is %f, max GPU allocate %f GB"%(avg_tokens, accuracy, torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024), flush=True)



if __name__ == "__main__":
    args = parse_config()
    main(args)

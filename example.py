from transformers import AutoTokenizer, TextStreamer, GenerationConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# load longheads model
from modeling_longheads import LlamaForCausalLM

longheads_config = {
    # chunk size setting for longheads
    'window_size':256,
    # the attention window length of longheads (atten_length should be smaller to model's pretrained length)
    'atten_length':4096,
    # during encoding phrase, we use this praram to begin streamingly encoding long context with chunk selection strategy
    'begin_selective_length':4096,
    # whether offload KV cache to cpu memory, if True longheads can generate to 128k+ context length
    'cpu_offload':False,
    # whether use batch_encoding for encoding phrase acceleration, if True more memory will be needed
    'batch_encoding':False,
    # the hyper param for batch encoding
    'encoding_batch_size':128,
}
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, **longheads_config,).cuda()



example_text = "What is Data Science?"
inputs = tokenizer(example_text, return_tensors='pt', add_special_tokens=False, return_token_type_ids=False)
input_ids = inputs['input_ids'].cuda()

with torch.no_grad():
    # A TextStreamer prints tokens as they're being generated
    streamer = TextStreamer(tokenizer)
    generated_tokens = model.generate(
        input_ids,
        generation_config=GenerationConfig(
            # use_cache=True is required, the rest can be changed up.
            use_cache=True,
            min_new_tokens=100,
            max_new_tokens=30_000,
        ),
        streamer=streamer,
    )
    # Decode the final generated text
    output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)




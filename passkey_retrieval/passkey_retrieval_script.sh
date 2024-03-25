MODEL_PATH="meta-llama/Llama-2-7b-hf"
NUM_TESTS=50

for TEST_K_TOKENS in 2 4 8 16;
do
    python3 passkey_retrieval/passkey_retrieval.py.py \
        --context_size 4096 \
        --base_model $MODEL_PATH \
        --test_k_tokens $TEST_K_TOKENS \
        --num_tests $NUM_TESTS 
done

import lightseq.inference as lsi
import numpy as np
import sys
import time
import torch


def generate_src_word(batch_size, vocab_size, max_length, pad_idx):
    memory_sequence_length = np.random.randint(
        low=1, high=max_length + 1, size=batch_size).astype(np.int32)
    max_length_idx = np.random.choice(range(batch_size), 1)
    memory_sequence_length[max_length_idx] = max_length
    data = []
    for i in range(batch_size):
        data.append(
            np.random.randint(
                low=3,
                high=vocab_size,
                size=memory_sequence_length[i],
                dtype=np.int64).tolist())

    src_word = [i + [pad_idx] * (max_length - len(i)) for i in data]
    return np.array(src_word, dtype=np.int32)


def test():
    np.random.seed(1245)
    batch_size = int(sys.argv[1])
    max_length = int(sys.argv[2])
    beam_size = int(sys.argv[3])
    vocab_size = 30000
    pad_idx = 0
    model = f"lightseq_transformer_bs_{beam_size}_len_{max_length}.pb"
    transformer = lsi.Transformer(
        model,
        batch_size)  # 32 is max batch size, it will decide GPU memory occupancy.
    test_input = generate_src_word(batch_size, vocab_size, max_length, pad_idx)
    print(test_input.shape)
    for i in range(100):
        # For warmup.
        if 50 == i:
            torch.cuda.synchronize()
            start = time.time()
        result = transformer.infer(test_input)
    torch.cuda.synchronize()
    end = (time.time() - start) / 50 * 1000
    print(
        f"LightseqResult: FP16, batch_size= {batch_size} , max_length= {max_length} , beam_size= {beam_size} , time= {round(end, 6)} ms/batch"
    )
    print(result)


if __name__ == "__main__":
    test()

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from attrdict import AttrDict
import argparse
import time

import paddle
import yaml
from pprint import pprint
from paddlenlp.ops import TransformerGenerator
from paddlenlp.utils.log import logger
from paddlenlp.data import Pad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/decoding.sample.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--decoding_lib",
        default="./build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--without_ft",
        action="store_true",
        help="Whether to use FasterTransformer to do predict. ")
    parser.add_argument(
        "--beam_size",
        type=int,
        required=True,
        help="The parameters for beam search.")
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        help="Max length of sequences deciding the size of position encoding table."
    )
    parser.add_argument("--max_out_len", type=int, required=True)
    parser.add_argument("--infer_batch_size", type=int, required=True)
    parser.add_argument("--beam_search_version", type=str, required=True)
    args = parser.parse_args()
    return args


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

    src_word = Pad(pad_idx)(data)
    return paddle.to_tensor(src_word, dtype="int64")


def do_predict(args):
    place = "gpu"
    paddle.set_device(place)

    # Define model
    transformer = TransformerGenerator(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        num_encoder_layers=0,
        num_decoder_layers=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len,
        use_ft=not args.without_ft,
        beam_search_version=args.beam_search_version,
        rel_len=args.use_rel_len,  # only works when using FT or beam search v2
        alpha=args.alpha,  # only works when using beam search v2
        diversity_rate=args.diversity_rate,  # only works when using FT
        use_fp16_decoding=args.use_fp16_decoding)  # only works when using FT

    # Set evaluate mode
    transformer.eval()

    src_word = generate_src_word(
        batch_size=args.infer_batch_size,
        vocab_size=args.src_vocab_size,
        max_length=args.max_length,
        pad_idx=args.bos_idx)
    print(src_word.shape)

    with paddle.no_grad():
        for i in range(100):
            # For warmup.
            if 50 == i:
                paddle.device.cuda.synchronize()
                start = time.time()
            transformer(src_word=src_word)
        paddle.device.cuda.synchronize()
        cost_time = round((time.time() - start) / 50 * 1000, 6)
        if not args.without_ft:
            logger.info(
                f'DecodingResult1: FT: {not args.without_ft}, {"FP16" if args.use_fp16_decoding else "FP32"}, beam_search_version: {args.beam_search_version}, infer_batch_size: {args.infer_batch_size}, seq_len: {args.max_out_len}, beam_size: {args.beam_size}, time:{cost_time} ms/batch'
            )
        else:
            logger.info(
                f'DecodingResult2: FT: {not args.without_ft}, beam_search_version: {args.beam_search_version}, infer_batch_size: {args.infer_batch_size}, seq_len: {args.max_out_len}, beam_size: {args.beam_size}, time:{cost_time} ms/batch'
            )


if __name__ == "__main__":
    np.random.seed(1245)
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    args.without_ft = ARGS.without_ft
    args.decoding_lib = ARGS.decoding_lib
    args.use_fp16_decoding = ARGS.use_fp16_decoding
    args.beam_size = ARGS.beam_size
    args.max_length = ARGS.max_length
    args.max_out_len = ARGS.max_out_len
    args.use_fp16_decoding = ARGS.use_fp16_decoding
    args.decoding_lib = ARGS.decoding_lib
    args.infer_batch_size = ARGS.infer_batch_size
    args.beam_search_version = ARGS.beam_search_version
    pprint(args)
    do_predict(args)

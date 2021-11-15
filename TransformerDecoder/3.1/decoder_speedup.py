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
import os
import argparse
import time
from pprint import pprint
import yaml
from attrdict import AttrDict
import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import TransformerModel, position_encoding_init
from paddlenlp.ops import FasterDecoder
from paddlenlp.utils.log import logger
from paddlenlp.data import Pad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./config/decoder.sample.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--decoder_lib",
        default="../../build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--use_fp16_decoder",
        action="store_true",
        help="Whether to use fp16 decoder to predict. ")
    parser.add_argument("--infer_batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, required=True)
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
                dtype=np.int64))

    word_pad = Pad(pad_idx)
    src_word = word_pad(data)

    return paddle.to_tensor(
        memory_sequence_length, dtype="int32"), paddle.to_tensor(
            src_word, dtype="int64")


def get_intput(args, transformer):
    mem_seq_lens, src_word = generate_src_word(
        batch_size=args.infer_batch_size,
        vocab_size=args.src_vocab_size,
        max_length=args.max_length,
        pad_idx=args.bos_idx)
    src_max_len = paddle.shape(src_word)[-1]
    src_slf_attn_bias = paddle.cast(
        src_word == args.bos_idx,
        dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
    src_slf_attn_bias.stop_gradient = True
    trg_src_attn_bias = src_slf_attn_bias
    src_pos = paddle.cast(
        src_word != args.bos_idx, dtype="int64") * paddle.arange(
            start=0, end=src_max_len)
    src_emb = transformer.src_word_embedding(src_word)
    src_pos_emb = transformer.src_pos_embedding(src_pos)
    src_emb = src_emb + src_pos_emb
    enc_input = F.dropout(
        src_emb, p=args.dropout, training=False) if args.dropout else src_emb
    enc_output = transformer.transformer.encoder(
        enc_input, src_mask=src_slf_attn_bias)
    dec_input = paddle.randn(
        shape=[args.infer_batch_size, 1, args.d_model], dtype='float32')
    caches = transformer.transformer.decoder.gen_cache(enc_output, do_zip=False)

    return dec_input, enc_output, mem_seq_lens, trg_src_attn_bias, caches


def do_predict(args):
    place = "gpu"
    paddle.set_device(place)
    paddle.seed(12345)
    np.random.seed(12345)

    print(f'{"=" * 20}TransformerModel{"=" * 20}')
    # Define model
    transformer = TransformerModel(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx)

    # Load the trained model
    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")
    model_dict = paddle.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))

    # To avoid a longer length than training, reset the size of position
    # encoding to max_length
    model_dict["src_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    model_dict["trg_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)

    transformer.load_dict(model_dict)

    # Set evaluate mode
    transformer.eval()

    # Generate data randomly
    dec_input, enc_output, mem_seq_lens, trg_src_attn_bias, caches = get_intput(
        args, transformer)
    logger.info(f'dec_input: {dec_input.shape}')
    logger.info(f'enc_output: {enc_output.shape}')
    logger.info(f'mem_seq_lens: {mem_seq_lens.shape}')
    with paddle.no_grad():
        for i in range(100):
            # For warmup.
            if 50 == i:
                paddle.device.cuda.synchronize()
                start = time.time()
            # For be same with FT decoder, you should remove last LayerNorm in paddle.nn.TransformerDecoder
            # https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/nn/layer/transformer.py#L1080
            pd_dec_output, caches = transformer.transformer.decoder(
                dec_input, enc_output, None, trg_src_attn_bias, caches)
        paddle.device.cuda.synchronize()
        end_1 = (time.time() - start) / 50 * 1000
        logger.info("Average test time for TransformerModel decoder is %f ms" %
                    (end_1))
    print(pd_dec_output)
    print(f'{"=" * 20}FasterDecoder{"=" * 20}')
    transformer = FasterDecoder(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        max_out_len=args.max_out_len,
        decoder_lib=args.decoder_lib,
        use_fp16_decoder=args.use_fp16_decoder)
    # Load checkpoint.
    transformer.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))
    # Set evaluate mode
    transformer.eval()

    dtype = 'float32'
    if args.use_fp16_decoder:
        dtype = 'float16'
        dec_input = paddle.cast(dec_input, dtype=dtype)
        enc_output = paddle.cast(enc_output, dtype=dtype)
    self_cache = paddle.zeros(
        shape=[
            args.num_decoder_layers, 2, 0, args.infer_batch_size, args.d_model
        ],
        dtype=dtype)
    mem_cache = paddle.zeros(
        shape=[
            args.num_decoder_layers, 2, args.infer_batch_size, args.max_length,
            args.d_model
        ],
        dtype=dtype)
    with paddle.no_grad():
        for i in range(100):
            # For warmup.
            if 50 == i:
                paddle.device.cuda.synchronize()
                start = time.time()
            ft_dec_output, self_cache, mem_cache = transformer.decoder(
                from_tensor=dec_input,
                memory_tensor=enc_output,
                mem_seq_len=mem_seq_lens,
                self_cache=self_cache,
                mem_cache=mem_cache)
        paddle.device.cuda.synchronize()
        end_2 = (time.time() - start) / 50 * 1000
        logger.info("Average test time for FasterDecoder decoder is %f ms" %
                    (end_2))
    print(ft_dec_output)
    if args.use_fp16_decoder:
        ft_dec_output = paddle.cast(ft_dec_output, dtype='float32')
    logger.info(
        f'max diff: {paddle.max(paddle.abs(ft_dec_output - pd_dec_output)).item()}'
    )
    logger.info(
        f'min diff: {paddle.min(paddle.abs(ft_dec_output - pd_dec_output)).item()}'
    )
    logger.info("Speed up is %f " % (end_1 / end_2))
    if args.use_fp16_decoder:
        logger.warning(
            f"FP16, batch_size= {args.infer_batch_size} , max_length= {args.max_length} , TransformerModel= {round(end_1, 6)} ms/batch, FasterDecoder= {round(end_2, 6)} ms/batch, Speedup= {round(end_1 / end_2, 6)}"
        )
    else:
        logger.warning(
            f"FP32, batch_size= {args.infer_batch_size} , max_length= {args.max_length} , TransformerModel= {round(end_1, 6)} ms/batch, FasterDecoder= {round(end_2, 6)} ms/batch, Speedup= {round(end_1 / end_2, 6)}"
        )


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
    args.decoder_lib = ARGS.decoder_lib
    args.use_fp16_decoder = ARGS.use_fp16_decoder
    args.infer_batch_size = ARGS.infer_batch_size
    args.max_length = ARGS.max_length
    pprint(args)

    do_predict(args)

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
from attrdict import AttrDict
import argparse
import time

import yaml
from pprint import pprint
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from paddlenlp.ops import FasterDecoder
from paddlenlp.utils.log import logger
from paddlenlp.ops import InferTransformerDecoder
from paddlenlp.utils.log import logger
from paddlenlp.transformers import WordEmbedding, PositionalEmbedding, position_encoding_init


class Model(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 bos_id=0,
                 eos_id=1,
                 max_out_len=256):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout
        self.max_out_len = max_out_len
        self.d_model = d_model

        self.src_word_embedding = WordEmbedding(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
        # print(self.src_word_embedding.word_embedding.weight)
        self.src_pos_embedding = PositionalEmbedding(
            emb_dim=d_model, max_length=max_length)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length)

        self.transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            activation="relu",
            normalize_before=True)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(x=x,
                                                  y=self.trg_word_embedding.word_embedding.weight,
                                                  transpose_y=True)
        else:
            self.linear = nn.Linear(
                in_features=d_model,
                out_features=trg_vocab_size,
                bias_attr=False)

    def forward(self, src_word):
        src_max_len = paddle.shape(src_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        src_slf_attn_bias.stop_gradient = True
        src_pos = paddle.cast(
            src_word != self.bos_id, dtype="int64") * paddle.arange(
                start=0, end=src_max_len)

        trg_src_attn_bias = src_slf_attn_bias

        src_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=self.training) if self.dropout else src_emb
        enc_output = self.transformer.encoder(
            enc_input, src_mask=src_slf_attn_bias)

        batch_size = enc_output.shape[0]
        end_token_tensor = paddle.full(
            shape=[batch_size, 1], fill_value=self.eos_id, dtype="int64")

        predict_ids = []
        log_probs = paddle.full(
            shape=[batch_size, 1], fill_value=0, dtype="float32")
        trg_word = paddle.full(
            shape=[batch_size, 1], fill_value=self.bos_id, dtype="int64")

        # init states (caches) for transformer
        caches = self.transformer.decoder.gen_cache(enc_output, do_zip=False)

        for i in range(args.max_out_len):
            trg_pos = paddle.full(
                shape=trg_word.shape, fill_value=i, dtype="int64")
            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            dec_output, caches = self.transformer.decoder(
                dec_input, enc_output, None, trg_src_attn_bias, caches)

            dec_output = paddle.reshape(
                dec_output, shape=[-1, dec_output.shape[-1]])

            logits = self.linear(dec_output)
            step_log_probs = paddle.log(F.softmax(logits, axis=-1))
            log_probs = paddle.add(x=step_log_probs, y=log_probs)
            scores = log_probs
            topk_scores, topk_indices = paddle.topk(x=scores, k=1)

            finished = paddle.equal(topk_indices, end_token_tensor)
            trg_word = topk_indices
            log_probs = topk_scores

            predict_ids.append(topk_indices)

            if paddle.all(finished).numpy():
                break

        predict_ids = paddle.stack(predict_ids, axis=0)
        finished_seq = paddle.transpose(predict_ids, [1, 2, 0])
        finished_scores = topk_scores

        return finished_seq, finished_scores


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


def get_op_cache_config(use_batch_major_op_cache, size_per_head, is_fp16):
    x = 8 if is_fp16 else 4
    use_batch_major_op_cache = True if use_batch_major_op_cache == True and \
                                       size_per_head % x == 0 \
                                    else False
    x = x if use_batch_major_op_cache else 1
    return use_batch_major_op_cache, x


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)
    paddle.seed(5678)
    # Define model
    transformer = Model(
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
        max_out_len=args.max_out_len, )

    # Load the trained model
    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")

    model_dict = paddle.load(
        os.path.join(args.init_from_params, "transformer.pdparams"),
        return_numpy=True)
    # To set weight[padding_idx] to 0.
    model_dict["trg_word_embedding.word_embedding.weight"][
        args.bos_idx] = [0] * args.d_model

    model_dict["src_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    model_dict["trg_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    transformer.load_dict(model_dict)

    # Set evaluate mode
    transformer.eval()

    # Generate src_word randomly
    src_word = paddle.randint(
        0,
        args.src_vocab_size,
        shape=[args.infer_batch_size, args.max_length],
        dtype='int64')
    print(src_word)
    print(f'{"=" * 20}TransformerModel{"=" * 20}')
    with paddle.no_grad():
        for i in range(1):
            # For warmup. 
            if 0 == i:
                start = time.time()
            pd_finished_seq, pd_finished_scores = transformer(src_word=src_word)
        print(pd_finished_seq, pd_finished_scores)
        end_1 = (time.time() - start) / 50 * 1000
        logger.info("Average test time for TransformerModel is %f ms" % (end_1))

    use_batch_major_op_cache = True
    size_per_head = args.d_model // args.n_head
    use_batch_major_op_cache, x = get_op_cache_config(
        use_batch_major_op_cache, size_per_head, args.use_fp16_decoder)
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
        use_fp16_decoder=args.use_fp16_decoder,
        use_batch_major_op_cache=use_batch_major_op_cache)

    # Load checkpoint.
    transformer.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))
    # Set evaluate mode
    transformer.eval()

    print(f'{"=" * 20}FasterDecoder{"=" * 20}')
    with paddle.no_grad():
        for i in range(1):
            # For warmup. 
            if 0 == i:
                start = time.time()
            ft_finished_seq, ft_finished_scores = transformer(src_word=src_word)
        print(ft_finished_seq, ft_finished_scores)
        end_2 = (time.time() - start) / 50 * 1000
        logger.info("Average test time for FasterDecoderis %f ms" % (end_2))

    logger.info(
        f'max diff finished_seq: {paddle.max(paddle.abs(pd_finished_seq - ft_finished_seq)).item()}'
    )
    logger.info(
        f'min diff finished_seq: {paddle.min(paddle.abs(pd_finished_seq - ft_finished_seq)).item()}'
    )
    logger.info(
        f'max diff finished_scores: {paddle.max(paddle.abs(pd_finished_scores - ft_finished_scores)).item()}'
    )
    logger.info(
        f'min diff finished_scores: {paddle.min(paddle.abs(pd_finished_scores - ft_finished_scores)).item()}'
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

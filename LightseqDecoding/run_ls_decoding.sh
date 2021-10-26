export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/for_pr/PaddleNLP_test_pr/:$PYTHONPATH
for beam_size in 1 4 ;
do
for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do
    echo ====================================================================
    echo beam_size:$beam_size, batch_size:$batch_size, seq_len:$seq_len
    echo ====================================================================
    cd /home/for_pr/PaddleNLP_test_pr/paddlenlp/ops/faster_transformer/
    # get paddlepaddle model with no encoder
    python3.7 sample/decoding_sample.py --max_length $seq_len
    cd -
    python3.7 paddle2lightseq_transformer.py $beam_size $seq_len
    python3.7 infer.py $batch_size $seq_len $beam_size
done
done
done
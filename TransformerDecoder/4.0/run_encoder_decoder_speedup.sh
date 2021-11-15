for infer_batch_size in 1 8 32 64 128 ;
do
for max_length in 32 64 128 ;
do
if [ "$precision" = "fp16" ]; then
    echo "Using fp16."
    python3.7 encoder_decoder_speedup.py --infer_batch_size ${infer_batch_size} --max_length ${max_length} --use_fp16_decoder
else
    echo "Using fp32"
    python3.7 encoder_decoder_speedup.py --infer_batch_size ${infer_batch_size} --max_length ${max_length}
fi
done
done
done

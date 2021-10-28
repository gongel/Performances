export CUDA_VISIBLE_DEVICES=4

for beam_search_version in v1 v2 ;
do
echo "########################################################"
echo ======================== With FT ========================
echo "########################################################"
for precision in fp32 fp16;
do
for beam_size in 1 4 ;
do
for infer_batch_size in 1 8 32 64 128 ;
do
for max_length in 32 64 128 ;
do
    echo ===============================================================================================================================
    echo batch_size:$infer_batch_size, seq_len:$max_length, beam_size:$beam_size, $precision ,beam_search_version:$beam_search_version
    echo ================================================================================================================================
if [ "$precision" = "fp16" ]; then
    echo "Using fp16."
    python3.7 -u encoder_decoding_sample.py --infer_batch_size ${infer_batch_size} --max_length ${max_length} --max_out_len ${max_length} --beam_size ${beam_size} --beam_search_version $beam_search_version --use_fp16_decoding
else
    echo "Using fp32"
    python3.7 -u encoder_decoding_sample.py --infer_batch_size ${infer_batch_size} --max_length ${max_length} --max_out_len ${max_length} --beam_size ${beam_size} --beam_search_version $beam_search_version
fi
done
done
done
done
done


echo "########################################################"
echo ======================== No FT ========================
echo "########################################################"
for beam_size in 1 4 ;
do
for infer_batch_size in 1 8 32 64 128 ;
do
for max_length in 32 64 128 ;
do
    echo =======================================================================================
    echo batch_size:$infer_batch_size, seq_len:$max_length, beam_size:$beam_size,beam_search_version:v1
    echo =======================================================================================
    python3.7 -u encoder_decoding_sample.py --infer_batch_size ${infer_batch_size} --max_length ${max_length} --max_out_len ${max_length} --beam_size ${beam_size} --beam_search_version v1 --without_ft
done
done
done
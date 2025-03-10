
CUDA_VISIBLE_DEVICES=7 python inference.py \
    --model_name ./model/google/flan-t5-large-train_r_aug_no_factual_nq\
    --input_dir ./mydata/nq/base/test.json\
    --output_dir ./mydata/analysis/nq/pray\
    --out_file_name flan-t5-large-train_r_aug_no_factual_nq_test-it_compressor


CUDA_VISIBLE_DEVICES=7 python eval.py \
    --target_model_name ./model/Llama-3_1-it \
    --input_dir ./mydata/analysis/nq/pray/flan-t5-large-train_r_aug_no_factual_nq_test-it_compressor.json \
    --output_dir ./mydata/analysis/nq/pray   \
    --out_file_name flan-t5-large-train_r_aug_no_factual_nq_test-it_eval\
    --dataset_name nq
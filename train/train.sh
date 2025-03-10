
export WANDB_DISABLED=true



WANDB_DISABLED=$WANDB_DISABLED CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 --master_port=3312 train.py\
    --model_name_or_path google/flan-t5-large \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --train_file ./mydata/nq/summarization/train_r_aug_no_factual.json\
    --validation_file ./mydata/nq/summarization/dev_r_aug_no_factual.json\
    --max_target_length 512 \
    --output_dir ./model/google/flan-t5-large-train_r_aug_no_factual_nq\
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 3 \
    --logging_first_step True \
    --max_eval_samples 10000 \
    --load_best_model_at_end

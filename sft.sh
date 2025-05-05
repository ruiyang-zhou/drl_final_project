export CUDA_VISIBLE_DEVICES=1
main_process_port=10011

#####--adapter_model  \
python /home/ryzhou/embd_model_sft/finetune_evaluate.py \
    --adapter_model linear \
    --dataset_version cb \
    --dataset_path /home/ryzhou/embd_model_sft \
    --model_version small-1.5 \
    --batch_size 8 \
    --epochs 1 \
    --modes train eval \
    --result_output_path /home/ryzhou/embd_model_sft/output
    
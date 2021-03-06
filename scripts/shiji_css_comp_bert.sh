# -*- coding: utf-8 -*-

data_dir=dataset/css/shiji
output_dir=output
config_path=configs/wcm_bert.json
bert_model=chinese_wcm_jt_pytorch

data_sign=shiji_css
task_name=BIO_cws
max_seq_len=128
train_batch=32
dev_batch=32
test_batch=32
learning_rate=5e-5
num_train_epochs=4
warmup=0.1
checkpoint=1000
output_model_name=shiji_css_ft_wcm_full_pytorch.bin

export CUDA_VISIBLE_DEVICES=3

python bin/run_bert_tagger.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_train \
--do_eval \
--train_batch_size ${train_batch} \
--dev_batch_size ${dev_batch} \
--test_batch_size ${test_batch} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--checkpoint ${checkpoint} \
--warmup_proportion ${warmup} \
--output_dir ${output_dir} \
--output_model_name ${output_model_name} \
--use_comp

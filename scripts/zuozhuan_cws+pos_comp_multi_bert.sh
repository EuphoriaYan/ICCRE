# -*- coding: utf-8 -*-

data_dir=dataset/cws+pos/zuozhuan
output_dir=output
config_path=configs/wcm_bert.json
bert_model=chinese_wcm_full_pytorch
device=cuda:1

data_sign=zuozhuan_pos
task_name=cws\&pos
max_seq_len=128
train_batch=32
dev_batch=32
test_batch=32
learning_rate=1e-4
num_train_epochs=10
warmup=0.1
checkpoint=500
output_model_name=zuozhuan_cws+pos_ft_wcm_full_pytorch.bin


python bin/run_bert_multi_tagger.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_train \
--do_eval \
--use_server \
--train_batch_size ${train_batch} \
--dev_batch_size ${dev_batch} \
--test_batch_size ${test_batch} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--checkpoint ${checkpoint} \
--warmup_proportion ${warmup} \
--output_dir ${output_dir} \
--output_model_name ${output_model_name} \
--device ${device}
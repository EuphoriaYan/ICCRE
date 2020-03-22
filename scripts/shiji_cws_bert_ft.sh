# -*- coding: utf-8 -*-

data_dir=dataset/cws/shiji
output_dir=output
config_path=configs/traditional_bert.json
bert_model=traditional_chinese_ft
device=cuda:2

data_sign=whitespace_cws
task_name=BIO_cws
max_seq_len=128
train_batch=32
dev_batch=32
test_batch=32
learning_rate=5e-5
num_train_epochs=10
warmup=0.1
checkpoint=1000
output_model_name=shiji_cws_ft_chinese_bert_pytorch.bin


python bin/run_bert_tagger.py \
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
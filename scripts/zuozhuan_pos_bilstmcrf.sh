# -*- coding: utf-8 -*-

data_dir=dataset/cws+pos/zuozhuan
output_dir=output
config_path=configs/traditional_bert.json
bert_model=traditional_chinese_jt

data_sign=zuozhuan_pos
task_name=pos
max_seq_len=128
batch_size=32
learning_rate=1e-3
num_train_epochs=15
warmup=0.1
checkpoint=1000

export CUDA_VISIBLE_DEVICES=0

python bin/run_lstm_tagger.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_train \
--do_eval \
--batch_size ${batch_size} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--checkpoint ${checkpoint} \
--warmup_proportion ${warmup} \
--output_dir ${output_dir} \
--export_model false
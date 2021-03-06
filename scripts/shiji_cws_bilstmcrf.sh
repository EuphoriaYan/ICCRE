# -*- coding: utf-8 -*-

data_dir=dataset/cws/shiji
output_dir=output
config_path=configs/bert.json
bert_model=chinese_L-12_H-768_A-12

data_sign=whitespace_cws
task_name=BIO_cws
max_seq_len=128
batch_size=32
learning_rate=1e-3
num_train_epochs=10
warmup=0.1
checkpoint=500
output_model_name=shiji_cws_ft_bilstm_crf.bin

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
--output_model_name ${output_model_name}
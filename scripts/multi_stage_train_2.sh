# -*- coding: utf-8 -*-

output_dir=output
config_path=configs/wcm_bert.json
bert_model=chinese_wcm_jt_pytorch
device=cuda:3
book_dir=dataset/cws_new
raw_data=dataset/cws_new/shiji.txt
seed=77777

task_name=BIO_cws
max_seq_len=128
test_batch=32
train_iterator=5

python bin/run_multi_stage_training.py \
--config_path ${config_path} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--seed ${seed} \
--max_seq_length ${max_seq_len} \
--test_batch_size ${test_batch} \
--output_dir ${output_dir} \
--device ${device} \
--train_iterator ${train_iterator} \
--book_dir ${book_dir} \
--raw_data ${raw_data} \
--do_train \
--do_eval \
--use_server

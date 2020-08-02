# -*- coding: utf-8 -*-

data_dir=dataset/cws_new
output_dir=output
config_path=configs/traditional_bert.json
bert_model=traditional_chinese_jt

data_sign=book_cws
task_name=BIO_cws
max_seq_len=128
test_batch=32

ckpt_name=shiji_cws_jt_chinese_bert_pytorch.bin

export CUDA_VISIBLE_DEVICES=1

python bin/run_ckpt_bert_tagger.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_eval \
--test_batch_size ${test_batch} \
--output_dir ${output_dir} \
--ckpt_name ${ckpt_name}

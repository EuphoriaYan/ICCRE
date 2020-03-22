# -*- coding: utf-8 -*-

data_dir=dataset/cws_new
output_dir=output
config_path=configs/wcm_bert.json
bert_model=chinese_wcm_ft_pytorch
device=cuda:1

data_sign=book_cws
task_name=BIO_cws
max_seq_len=128
test_batch=32

ckpt_name=shiji_cws_ft_wcm_full_pytorch.bin

python bin/run_ckpt_bert_tagger.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_eval \
--use_server \
--test_batch_size ${test_batch} \
--output_dir ${output_dir} \
--ckpt_name ${ckpt_name} \
--device ${device}

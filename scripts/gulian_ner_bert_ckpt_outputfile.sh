
export CUDA_VISIBLE_DEVICES=0

python bin/run_ckpt_bert_tagger.py \
--data_sign gulian_ner \
--config_path configs/traditional_bert.json \
--data_dir dataset/gulian_txt \
--bert_model bert_daizhige \
--use_crf \
--task_name ner \
--max_seq_length 128 \
--do_eval \
--test_batch_size 64 \
--output_dir output \
--ckpt_name gulian_ner_bert_daizhige.bin \
--output_file \

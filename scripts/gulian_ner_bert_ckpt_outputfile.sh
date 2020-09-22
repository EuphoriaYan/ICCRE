
export CUDA_VISIBLE_DEVICES=0

python bin/run_ckpt_bert_tagger.py \
--data_sign gulian_test_ner \
--config_path configs/bert.json \
--data_dir dataset/gulian_txt \
--bert_model bert_daizhige \
--use_crf \
--task_name ner \
--max_seq_length 128 \
--do_eval \
--test_batch_size 64 \
--output_dir output \
--ckpt_name zztj_gulian_ner_bert_cpoac_dzg+diaolong+gl+zztj_2M_Kdev3.bin \
--output_file \

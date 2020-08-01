#!/bin/bash
while [ 1 ]
do
    exist=$(ps -p 7366)
    if [ "" == "$exist" ]; then
        nohup sh scripts/gulian_ner_bert.sh > output/gulian_ner_bert.log &
        exit
    fi
    usleep 1000
done &
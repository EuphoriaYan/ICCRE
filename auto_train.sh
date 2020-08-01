while True
do
    exist=`ps -p 7366`
    if [ "" == "$exist" ];
    then
        nohup sh scripts/
    fi
    usleep 1000
done &
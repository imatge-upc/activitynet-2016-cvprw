nohup srun -w c8 -c2 --mem 100000 --gres=gpu:1 python test_recurrent.py > logs/recurrent/test.log 2>&1 &
tail -f logs/recurrent/test.log

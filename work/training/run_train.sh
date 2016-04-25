nohup srun -w c8 --mincpus=10 --mem 60000 --gres=gpu:1 python train_finetune.py > train.log 2>&1 &
tail -f train.log

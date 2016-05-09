nohup srun -c2 --mem 30000 --gres=gpu:1 python extract_predictions.py > logs/predictions.log 2>&1 &
tail -f logs/predictions.log

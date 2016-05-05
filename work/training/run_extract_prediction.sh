nohup srun -w c6 -c2 --mem 30000 --gres=gpu:1 python validate_recurrent.py > logs/recurrent/extract_predictions.log 2>&1 &
tail -f logs/recurrent/extract_predictions.log

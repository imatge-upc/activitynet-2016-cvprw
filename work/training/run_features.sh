nohup srun -w c6 -c8 --mem 50000 --gres=gpu:1 python extract_features.py 0 6600 > logs/extract_features_5-4.log 2>&1 &
#nohup srun -w c6 -c8 --mem 50000 --gres=gpu:1 python extract_features.py 6600 20000 > logs/extract_features_5-2.log 2>&1 &
tail -f logs/extract_features_5-4.log

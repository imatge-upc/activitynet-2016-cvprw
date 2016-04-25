nohup srun -w c8 -c8 --mem 40000 --gres=gpu:1 python extract_features.py > logs/extract_features_2.log 2>&1 &
tail -f logs/extract_features_2.log

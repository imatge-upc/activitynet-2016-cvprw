# THEANO_FLAGS='base_compiledir=~/.theano/features1' nohup srun -c8 --mem 50000 --gres=gpu:1 python extract_features.py 0 10000 > logs/extract_features/extracting-1.log 2>&1 &
# THEANO_FLAGS='base_compiledir=~/.theano/features2' nohup srun -c8 --mem 50000 --gres=gpu:1 python extract_features.py 10000 20000 > logs/extract_features/extracting-2.log 2>&1 &
# tail -f logs/extract_features/extracting-1.log

# nohup srun -c8 --mem 50000 --gres=gpu:1 python extract_features.py 0 10000 > logs/extracting-3.log 2>&1 &
# tail -f logs/extracting-3.log

# nohup srun -c8 --mem 50000 --gres=gpu:1 python extract_features.py 0 10000 > logs/extracting-4.log 2>&1 &
# tail -f logs/extracting-4.log

# nohup srun -c8 --mem 50000 --gres=gpu:1 python extract_features.py 0 10000 > logs/extracting-5.log 2>&1 &
# tail -f logs/extracting-5.log

nohup srun -c8 --mem 50000 --gres=gpu:1 python extract_features.py 0 10000 > logs/extracting-6.log 2>&1 &
tail -f logs/extracting-6.log

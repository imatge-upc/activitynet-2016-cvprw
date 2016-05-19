experiment=$(ls -l logs/ | grep log | wc -l)
echo "Running experiment #$experiment for Activity Classification Neural Network"

nohup srun -u -c2 --mem 75G --gres=gpu:1 python train.py > logs/experiment_$experiment.log 2>&1 &
tail -f logs/experiment_$experiment.log

# srun -u -p fast --mem 30G python train.py

# experiment=$(ls -l logs/ | grep .v2.log | wc -l)
# echo "Running experiment #$experiment for Activity Classification Neural Network v2"
#
# nohup srun -u -c2 --mem 80G python -m memory_profiler train_2.py > logs/experiment_$experiment.v2.log 2>&1 &
# tail -f logs/experiment_$experiment.v2.log

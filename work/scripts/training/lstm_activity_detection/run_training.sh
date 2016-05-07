experiment=$(ls -l logs/ | grep log | wc -l)
echo "Running experiment #$experiment for Activity Detection Neural Network"

nohup srun -c2 --mem 130000 --gres=gpu:1 python train.py > logs/experiment_$experiment.log 2>&1 &
tail -f logs/experiment_$experiment.log

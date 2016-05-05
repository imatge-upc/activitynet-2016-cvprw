experiment=$(ls -l logs/recurrent/ | grep log | wc -l)
echo Running experiment \#$experiment

nohup srun -w c8 -c2 --mem 120000 --gres=gpu:1 python train_recurrent.py > logs/recurrent/experiment_$experiment.log 2>&1 &
tail -f logs/recurrent/experiment_$experiment.log

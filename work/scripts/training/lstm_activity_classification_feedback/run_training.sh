experiment=$(ls -l logs/ | grep log | wc -l)
echo "Running experiment #$experiment for Activity Classification Neural Network with feedback"

# srun -u -c2 --mem 20000 -p fast python train.py

THEANO_FLAGS='base_compiledir=~/.theano/classification' nohup srun -u -c2 --mem 10G --gres=gpu:1 python train.py &
tail -f logs/experiment_$experiment.log

python fsbbt_resnet.py --dataset cifar100 --studentdatatype subset --teacherdatatype original --cvaedatatype subset --cvaemethod input_cond --cvaelatent 2 --cvaebatchsize 64 --cvaeepochs 600 --balance 0 --batchsize 16 --epochs 200 --nogen 0 --sampling hybrid --augmentation standard --mxalpha 1 --nomixup 40000 --loss ce --round 2 --threshold 0.05 --run 1 --gpu -1

python cvae_alexnet.py --dataset cifar100 --cvaedatatype subset --cvaemethod input_cond --teacherdatatype original --latent 2 --batchsize 64 --epochs 600 --gpu -1
python cvae_resnet.py --dataset cifar100 --cvaedatatype subset --cvaemethod input_cond --teacherdatatype original --latent 2 --batchsize 64 --epochs 600 --gpu -1

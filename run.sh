export WORLD_SIZE=2
export MASTER_PORT=$(python -S -c "import random; print(random.randrange(10000, 19000))")
export OMP_NUM_THREADS=$((128/$WORLD_SIZE))
export MASTER_ADDR=$HOSTNAME
ROOT='/path/to/imagenet/ffcv/folder'

bs=$((1024/$WORLD_SIZE))
echo $bs

wandb_id=$(python -c "import wandb; print(wandb.util.generate_id())")
total="50ep"
ssr=2
surgery="90ep"
bs_multiplier=2
#  a checkpoint will be dumped at the surgery time which will not be deleted by the CheckpointSaver callback
composer -n $WORLD_SIZE main.py --data $ROOT --train_bs $bs --test_bs $bs --epochs $total --scale_schedule_ratio $ssr --surgery_time $surgery --bs_multiplier $bs_multiplier --name total[${ssr}x${total}]_surgery[${surgery},bs${bs_multiplier}x]_$wandb_id --id $wandb_id --ckpt /path/to/dumped/ckpt.pt
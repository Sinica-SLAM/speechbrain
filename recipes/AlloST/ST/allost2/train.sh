python -m torch.distributed.launch \
    --nproc_per_node=2 train.py hparams/transformer.yaml \
    --distributed_launch --distributed_backend='nccl' \
    --find_unused_parameters
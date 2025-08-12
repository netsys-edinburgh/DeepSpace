import torch
import torch.distributed as dist
import os

def init_process(rank, size):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6027'
    dist.init_process_group('gloo', rank=rank, world_size=size)

def main(rank, size):
    init_process(rank, size)
    # Example tensor operation
    tensor = torch.zeros(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor[0])

if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

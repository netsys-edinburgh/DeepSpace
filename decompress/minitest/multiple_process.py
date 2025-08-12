import torch
import torch.multiprocessing as mp

def worker(rank, a_tensor):
    print(f"Worker {rank} received tensor: {a_tensor}")
    a_tensor += 1  # Modify the tensor
    print(f"Worker {rank} modified tensor: {a_tensor}")

def main():
    # Set the start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    # Create a shared tensor
    original_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)

    processes = []
    num_processes = 2  # Define the number of processes

    for rank in range(num_processes):
        # Create a Process
        p = mp.Process(target=worker, args=(rank, original_tensor))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Final tensor in main process:", original_tensor)

if __name__ == '__main__':
    main()

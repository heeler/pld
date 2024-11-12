import torch


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"{torch.cuda.device_count()} devices:")
        for i in range(torch.cuda.device_count()):
            print(f"\t device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available")

    print("cuda_check_finished.")
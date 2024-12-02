import os
import torch
import sys
import subprocess

def comprehensive_gpu_diagnostic():
    """
    Perform comprehensive GPU diagnostics
    """
    print("=" * 50)
    print("COMPREHENSIVE GPU DIAGNOSTIC")
    print("=" * 50)
    
    # Basic PyTorch CUDA Check
    print("\n1. PyTorch CUDA Availability")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    # CUDA Devices
    print("\n2. CUDA Device Information")
    print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        # Detailed GPU Information
        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i} Details:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            
            # Memory Information
            print("  Memory Information:")
            print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"    Allocated Memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"    Cached Memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    # System-level CUDA and Driver Check
    print("\n3. System CUDA Information")
    try:
        cuda_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        print("NVCC Version:\n", cuda_version)
    except Exception as e:
        print("Could not retrieve NVCC version:", e)
    
    # nvidia-smi output
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode('utf-8')
        print("\n4. nvidia-smi Output Excerpt:")
        print(nvidia_smi[:500] + "..." if len(nvidia_smi) > 500 else nvidia_smi)
    except Exception as e:
        print("Could not retrieve nvidia-smi output:", e)
    
    print("\n5. Environment Variables")
    cuda_related_vars = [var for var in os.environ if 'CUDA' in var or 'GPU' in var]
    for var in cuda_related_vars:
        print(f"{var}: {os.environ.get(var)}")

def test_cuda_tensor_operations():
    """
    Test basic CUDA tensor operations
    """
    print("\n=" * 50)
    print("CUDA TENSOR OPERATION TEST")
    print("=" * 50)
    
    try:
        # Create a tensor on GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.rand(5, 3, device=device)
        print("Tensor created on GPU:")
        print(x)
        
        # Perform a simple operation
        y = x * 2
        print("\nTensor multiplication successful")
        print(y)
    
    except Exception as e:
        print("Error in CUDA tensor operations:", e)

if __name__ == "__main__":
    comprehensive_gpu_diagnostic()
    test_cuda_tensor_operations()
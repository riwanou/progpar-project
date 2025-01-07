static __global__ void kernel_cuda_naive() {
  size_t xl = threadIdx.x, yl = threadIdx.y;
  size_t xg = blockDim.x * blockIdx.x + threadIdx.x;
  size_t yg = blockDim.y * blockIdx.y + threadIdx.y;
}


#include <string>

#include "SimulationNBodyCudaOptim3.hpp"

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                 \
            exit(err);                                                                                                 \
        }                                                                                                              \
    }

// flops = (n² / 1536) * (8 + (1536 * 39)) + 17 * n
__global__ void kernel_cuda_optim2(float *in_qx, float *in_qy, float *in_qz, float *in_m, accAoS_t<float> *outAccelerations,
                                   const unsigned int nbBodies, const float soft, const float G, const int offset)
{
    const int sizePass = 1536;
    const int nbPass = (nbBodies + sizePass - 1) / sizePass; // 3 flops

    static __shared__ float shBodies_x[sizePass];
    static __shared__ float shBodies_y[sizePass];
    static __shared__ float shBodies_z[sizePass];
    static __shared__ float shBodies_m[sizePass];

    const int iBody = (blockDim.x * blockIdx.x + threadIdx.x) * 2; // 3 flops
    const int iBody1 = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + 1; // 4 flops
    const float softSquared = soft * soft; // 1 flop

    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float4 q = make_float4(in_qx[iBody], in_qy[iBody], in_qz[iBody], 0.0f);
    float4 q1 = make_float4(in_qx[iBody1], in_qy[iBody1], in_qz[iBody1], 0.0f);

    // shared memory is too small to contains all the bodies
    // acumulate the acceleration in multiple passes
    // flops = (n / 1536) * (8 + (1536 * 39))
    for (int pass = 0; pass < nbPass; pass++) {
        const int startIdx = pass * sizePass; // 1 flop
        const int endIdx = min((pass + 1) * sizePass, nbBodies); // 3 flops
        unsigned int shIdx = 0;

        // load in shared memory
        shBodies_x[threadIdx.x] = in_qx[startIdx + threadIdx.x]; // 1 flop
        shBodies_y[threadIdx.x] = in_qy[startIdx + threadIdx.x]; // 1 flop
        shBodies_z[threadIdx.x] = in_qz[startIdx + threadIdx.x]; // 1 flop
        shBodies_m[threadIdx.x] = in_m[startIdx + threadIdx.x]; // 1 flop
        shBodies_x[threadIdx.x + 768] = in_qx[startIdx + threadIdx.x + 768]; // 3 flops
        shBodies_y[threadIdx.x + 768] = in_qy[startIdx + threadIdx.x + 768]; // 3 flops
        shBodies_z[threadIdx.x + 768] = in_qz[startIdx + threadIdx.x + 768]; // 3 flops
        shBodies_m[threadIdx.x + 768] = in_m[startIdx + threadIdx.x + 768]; // 3 flops
        __syncthreads();

        // flops = 1536 * 39
        for (int jBody = startIdx; jBody < endIdx; jBody++) {
            float4 shBody = make_float4(shBodies_x[shIdx], shBodies_y[shIdx], shBodies_z[shIdx], shBodies_m[shIdx]);

            float rijx = shBody.x - q.x; // 1 flop
            float rijy = shBody.y - q.y; // 1 flop
            float rijz = shBody.z - q.z; // 1 flop

            float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared; // 6 flops
            float revSqrt = rsqrtf(rijSquared); // 1 flop
            float rsqrt3 = revSqrt * revSqrt * revSqrt; // 2 flops
            float ai = shBody.w * rsqrt3; // 1 flop

            a.x += ai * rijx; // 2 flops
            a.y += ai * rijy; // 2 flops
            a.z += ai * rijz; // 2 flops

            rijx = shBody.x - q1.x; // 1 flop
            rijy = shBody.y - q1.y; // 1 flop
            rijz = shBody.z - q1.z; // 1 flop

            rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared; // 6 flops
            revSqrt = rsqrtf(rijSquared); // 1 flop
            rsqrt3 = revSqrt * revSqrt * revSqrt; // 2 flops
            ai = shBody.w * rsqrt3; // 1 flop

            a1.x += ai * rijx; // 2 flops
            a1.y += ai * rijy; // 2 flops
            a1.z += ai * rijz; // 2 flops

            shIdx++; // 1 flop
        }

        __syncthreads();
    }

    outAccelerations[iBody].ax = a.x * G; // 1 flop
    outAccelerations[iBody].ay = a.y * G; // 1 flop
    outAccelerations[iBody].az = a.z * G; // 1 flop
    outAccelerations[iBody1].ax = a1.x * G; // 1 flop
    outAccelerations[iBody1].ay = a1.y * G; // 1 flop
    outAccelerations[iBody1].az = a1.z * G; // 1 flop
}

SimulationNBodyCudaOptim3::SimulationNBodyCudaOptim3(const unsigned long nBodies, const std::string &scheme,
                                                     const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    // 2048 is shared memory size, corresponding to the number of passes.
    // flops = (n² / 1536) * (8 + (1536 * 39)) + 17 * n
    const float N = this->getBodies().getN();
    this->flopsPerIte = (N * N / 1536) * (8 + (1536 * 39)) + 17 * N;
    this->accelerations.resize(this->getBodies().getN());
    
    CUDA_CHECK(cudaMalloc(&cudaAccelerations, this->getBodies().getN() * sizeof(accAoS_t<float>)));
    CUDA_CHECK(cudaMalloc(&cuda_qx, this->getBodies().getN() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_qy, this->getBodies().getN() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_qz, this->getBodies().getN() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_m, this->getBodies().getN() * sizeof(float)));
}

void SimulationNBodyCudaOptim3::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    dim3 block(768);
    int nbBlocks = (this->getBodies().getN() + block.x - 1) / block.x;
    nbBlocks = (nbBlocks + 1) / 2;
    dim3 grid(nbBlocks);

    CUDA_CHECK(cudaMemcpy(this->cuda_qx, d.qx.data(),
                          this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->cuda_qy, d.qy.data(),
                          this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->cuda_qz, d.qz.data(),
                          this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->cuda_m, d.m.data(),
                          this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice));

    kernel_cuda_optim2<<<grid, block>>>(this->cuda_qx, this->cuda_qy, this->cuda_qz, this->cuda_m, this->cudaAccelerations, this->getBodies().getN(), this->soft,
                                        this->G, nbBlocks * block.x);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(this->accelerations.data(), this->cudaAccelerations,
                          this->getBodies().getN() * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost));
}

void SimulationNBodyCudaOptim3::computeOneIteration()
{
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

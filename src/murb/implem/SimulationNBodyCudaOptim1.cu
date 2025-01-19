#include <string>

#include "SimulationNBodyCudaOptim1.hpp"

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                 \
            exit(err);                                                                                                 \
        }                                                                                                              \
    }

// flops = (n² / 2048) * (8 + (2048 * 21)) + 6 * n
__global__ void kernel_cuda_optim1(cudaPackedAoS_t<float> *inBodies, accAoS_t<float> *outAccelerations, const unsigned int nbBodies,
                                   const float soft, const float G)
{
    const unsigned int sizePass = 2048;
    const unsigned int nbPass = (nbBodies + sizePass - 1) / sizePass; // 3 flops

    extern __shared__ cudaPackedAoS_t<float> shBodies[sizePass];
    const unsigned int iBody = blockDim.x * blockIdx.x + threadIdx.x; // 2 flops
    if (iBody >= nbBodies) return;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    const float softSquared = soft * soft; // 1 flop

    float qx_i = inBodies[iBody].qx;
    float qy_i = inBodies[iBody].qy;
    float qz_i = inBodies[iBody].qz;

    // shared memory is too small to contains all the bodies
    // acumulate the acceleration in multiple passes
    // flops = (n / 2048) * (8 + (2048 * 21))
    for (uint pass = 0; pass < nbPass; pass++) {
        const unsigned int startIdx = pass * sizePass; // 1 flop
        const unsigned int endIdx = min((pass + 1) * sizePass, nbBodies);  // 3 flops
        unsigned int shIdx = 0;

        // load in shared memory
        shBodies[threadIdx.x] = inBodies[startIdx + threadIdx.x]; // 1 flop
        shBodies[threadIdx.x + 1024] = inBodies[startIdx + threadIdx.x + 1024]; // 3 flops
        __syncthreads();

        // flops = 2048 * 21
        for (unsigned int jBody = startIdx; jBody < endIdx; jBody++) {
            const float rijx = shBodies[shIdx].qx - qx_i; // 1 flop
            const float rijy = shBodies[shIdx].qy - qy_i; // 1 flop
            const float rijz = shBodies[shIdx].qz - qz_i; // 1 flop

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared; // 6 flops
            const float revSqrt = rsqrtf(rijSquared); // 1 flop
            const float rsqrt3 = revSqrt * revSqrt * revSqrt; // 2 flops
            const float ai = G * shBodies[shIdx].m * rsqrt3; // 2 flops

            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops

            shIdx++; // 1 flop
        }

        __syncthreads();
    }

    outAccelerations[iBody].ax = ax;
    outAccelerations[iBody].ay = ay;
    outAccelerations[iBody].az = az;
}

SimulationNBodyCudaOptim1::SimulationNBodyCudaOptim1(const unsigned long nBodies, const std::string &scheme,
                                                     const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    // 2048 is shared memory size, corresponding to the number of passes.
    // flops = (n² / 2048) * (8 + (2048 * 21)) + 6 * n
    const float N = this->getBodies().getN();
    this->flopsPerIte = (N * N / 2048) * (8 + (2048 * 21)) + 6 * N;
    this->accelerations.resize(this->getBodies().getN());
    this->packedBodies.resize(this->getBodies().getN());

    CUDA_CHECK(cudaMalloc(&cudaAccelerations, this->getBodies().getN() * sizeof(accAoS_t<float>)));
    CUDA_CHECK(cudaMalloc(&cudaBodies, this->getBodies().getN() * sizeof(cudaPackedAoS_t<float>)));
}

void SimulationNBodyCudaOptim1::initIteration()
{
    std::vector<dataAoS_t<float>> bodies = this->getBodies().getDataAoS();

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        dataAoS_t<float> body = bodies[iBody];
        this->packedBodies[iBody].qx = body.qx;
        this->packedBodies[iBody].qy = body.qy;
        this->packedBodies[iBody].qz = body.qz;
        this->packedBodies[iBody].m = body.m;
    }
}

void SimulationNBodyCudaOptim1::computeBodiesAcceleration()
{
    dim3 block(1024);
    dim3 grid((this->getBodies().getN() + block.x - 1) / block.x);

    CUDA_CHECK(cudaMemcpy(this->cudaBodies, this->packedBodies.data(), this->getBodies().getN() * sizeof(cudaPackedAoS_t<float>),
                          cudaMemcpyHostToDevice));

    kernel_cuda_optim1<<<grid, block>>>(
        this->cudaBodies, this->cudaAccelerations, this->getBodies().getN(), this->soft, this->G);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(this->accelerations.data(), this->cudaAccelerations,
                          this->getBodies().getN() * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost));
}

void SimulationNBodyCudaOptim1::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

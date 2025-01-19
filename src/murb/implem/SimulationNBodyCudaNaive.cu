#include <cmath>
#include <string>

#include "SimulationNBodyCudaNaive.hpp"

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                 \
            exit(err);                                                                                                 \
        }                                                                                                              \
    }

// flops = n² * 20 + n
__global__ void kernel_cuda_naive(dataAoS_t<float> *inBodies, accAoS_t<float> *outAccelerations, const size_t nbBodies,
                                  const float soft, const float G)
{
    size_t iBody = blockDim.x * blockIdx.x + threadIdx.x;
    if (iBody > nbBodies) return;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    float softSquared = soft * soft; // 1 flop

    float qx = inBodies[iBody].qx;
    float qy = inBodies[iBody].qy;
    float qz = inBodies[iBody].qz;

    // flops = n * 20
    for (int jBody = 0; jBody < nbBodies; jBody++) {
        const float rijx = inBodies[jBody].qx - qx; // 1 flop
        const float rijy = inBodies[jBody].qy - qy; // 1 flop
        const float rijz = inBodies[jBody].qz - qz; // 1 flop

        const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared; // 6 flops
        const float revSqrt = rsqrtf(rijSquared); // 1 flop
        const float rsqrt3 = revSqrt * revSqrt * revSqrt; // 2 flops
        const float ai = G * inBodies[jBody].m * rsqrt3; // 2 flops

        ax += ai * rijx; // 2 flops
        ay += ai * rijy; // 2 flops
        az += ai * rijz; // 2 flops
    }

    outAccelerations[iBody].ax = ax;
    outAccelerations[iBody].ay = ay;
    outAccelerations[iBody].az = az;
}

SimulationNBodyCudaNaive::SimulationNBodyCudaNaive(const unsigned long nBodies, const std::string &scheme,
                                                   const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    const float N = this->getBodies().getN();
    this->flopsPerIte = (20.f * N * N) + N;
    this->accelerations.resize(this->getBodies().getN());

    CUDA_CHECK(cudaMalloc(&cudaAccelerations, this->getBodies().getN() * sizeof(accAoS_t<float>)));
    CUDA_CHECK(cudaMalloc(&cudaBodies, this->getBodies().getN() * sizeof(dataAoS_t<float>)));
}

void SimulationNBodyCudaNaive::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    dim3 block(1024);
    dim3 grid((this->getBodies().getN() + block.x - 1) / block.x);

    CUDA_CHECK(cudaMemcpy(this->cudaBodies, d.data(), this->getBodies().getN() * sizeof(dataAoS_t<float>),
                          cudaMemcpyHostToDevice));

    kernel_cuda_naive<<<grid, block>>>(this->cudaBodies, this->cudaAccelerations, this->getBodies().getN(), this->soft,
                                       this->G);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(this->accelerations.data(), this->cudaAccelerations,
                          this->getBodies().getN() * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost));
}

void SimulationNBodyCudaNaive::computeOneIteration()
{
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

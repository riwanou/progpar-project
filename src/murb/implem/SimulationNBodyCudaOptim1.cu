#include <cmath>
#include <string>

#include "SimulationNBodyCudaOptim1.hpp"

#define CUDA_CHECK(call) {                                            \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",                 \
                cudaGetErrorString(err), __FILE__, __LINE__);        \
        exit(err);                                                   \
    }                                                                \
}

__global__ void kernel_cuda_naive(dataAoS_t<float> *inBodies, accAoS_t<float> *outAccelerations, 
                                  const size_t nbBodies, const float soft, const float G) {
    size_t iBody = blockDim.x * blockIdx.x + threadIdx.x;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    float softSquared = soft * soft;

    float qx = inBodies[iBody].qx;
    float qy = inBodies[iBody].qy;
    float qz = inBodies[iBody].qz;

    for (int jBody = 0; jBody < nbBodies; jBody++) {
        const float rijx = inBodies[jBody].qx - qx; 
        const float rijy = inBodies[jBody].qy - qy;
        const float rijz = inBodies[jBody].qz - qz;

        const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared;
        const float revSqrt = rsqrt(rijSquared);
        const float rsqrt3 = revSqrt * revSqrt * revSqrt;
        const float ai = G * inBodies[jBody].m * rsqrt3;

        ax += ai * rijx;
        ay += ai * rijy;
        az += ai * rijz;
    }

    outAccelerations[iBody].ax = ax;
    outAccelerations[iBody].ay = ay;
    outAccelerations[iBody].az = az;
}

SimulationNBodyCudaOptim1::SimulationNBodyCudaOptim1(const unsigned long nBodies, const std::string &scheme, const float soft,
                                         const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = (20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN()) + (9.0f * (float)this->getBodies().getN());
    this->accelerations.resize(this->getBodies().getN());

    CUDA_CHECK(cudaMalloc(&cudaAccelerations, this->getBodies().getN() * sizeof(accAoS_t<float>)));
    CUDA_CHECK(cudaMalloc(&cudaBodies, this->getBodies().getN() * sizeof(dataAoS_t<float>)));
}

void SimulationNBodyCudaOptim1::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    dim3 block(512);
    dim3 grid((this->getBodies().getN() + block.x - 1) / block.x);

    CUDA_CHECK(cudaMemcpy(this->cudaBodies, d.data(), 
                          this->getBodies().getN() * sizeof(dataAoS_t<float>), cudaMemcpyHostToDevice));

    kernel_cuda_naive<<<grid, block>>>(this->cudaBodies, this->cudaAccelerations, this->getBodies().getN(), this->soft, this->G);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(this->accelerations.data(), this->cudaAccelerations, 
                          this->getBodies().getN() * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost));
}

void SimulationNBodyCudaOptim1::computeOneIteration()
{
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

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

__global__ void kernel_cuda_optim1(dataAoS_t<float> *inBodies, accAoS_t<float> *outAccelerations, const unsigned int nbBodies,
                                   const float soft, const float G)
{
    const unsigned int sizePass = 1024;
    const unsigned int nbPass = (nbBodies + sizePass - 1) / sizePass;

    extern __shared__ dataAoS_t<float> shBodies[sizePass];
    const unsigned int iBody = blockDim.x * blockIdx.x + threadIdx.x;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    const float softSquared = soft * soft;

    float qx_i = inBodies[iBody].qx;
    float qy_i = inBodies[iBody].qy;
    float qz_i = inBodies[iBody].qz;

    // shared memory is too small to contains all the bodies
    // acumulate the acceleration in multiple passes
    for (uint pass = 0; pass < nbPass; pass++) {
        const unsigned int startIdx = pass * sizePass;
        const unsigned int endIdx = min((pass + 1) * sizePass, nbBodies); 
        unsigned int shIdx = 0;

        // load in shared memory
        shBodies[threadIdx.x] = inBodies[startIdx + threadIdx.x];
        __syncthreads();

        for (unsigned int jBody = startIdx; jBody < endIdx; jBody++) {
            const float rijx = shBodies[shIdx].qx - qx_i;
            const float rijy = shBodies[shIdx].qy - qy_i;
            const float rijz = shBodies[shIdx].qz - qz_i;

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared;
            const float revSqrt = rsqrt(rijSquared);
            const float rsqrt3 = revSqrt * revSqrt * revSqrt;
            const float ai = G * shBodies[shIdx].m * rsqrt3;

            ax += ai * rijx;
            ay += ai * rijy;
            az += ai * rijz;

            shIdx++;
        }
    }

    outAccelerations[iBody].ax = ax;
    outAccelerations[iBody].ay = ay;
    outAccelerations[iBody].az = az;
}

SimulationNBodyCudaOptim1::SimulationNBodyCudaOptim1(const unsigned long nBodies, const std::string &scheme,
                                                     const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = (20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN()) +
                        (9.0f * (float)this->getBodies().getN());
    this->accelerations.resize(this->getBodies().getN());

    CUDA_CHECK(cudaMalloc(&cudaAccelerations, this->getBodies().getN() * sizeof(accAoS_t<float>)));
    CUDA_CHECK(cudaMalloc(&cudaBodies, this->getBodies().getN() * sizeof(dataAoS_t<float>)));
}

void SimulationNBodyCudaOptim1::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    dim3 block(1024);
    dim3 grid((this->getBodies().getN() + block.x - 1) / block.x);

    CUDA_CHECK(cudaMemcpy(this->cudaBodies, d.data(), this->getBodies().getN() * sizeof(dataAoS_t<float>),
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
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

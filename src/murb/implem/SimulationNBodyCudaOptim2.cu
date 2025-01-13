#include <string>

#include "SimulationNBodyCudaOptim2.hpp"

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                 \
            exit(err);                                                                                                 \
        }                                                                                                              \
    }

__global__ void kernel_cuda_optim2(cudaPackedAoS_t<float> *inBodies, accAoS_t<float> *outAccelerations,
                                   const unsigned int nbBodies, const float soft, const float G, const int offset)
{
    const int sizePass = 1536;
    const int nbPass = (nbBodies + sizePass - 1) / sizePass;

    static __shared__ cudaPackedAoS_t<float> shBodies[sizePass];

    const int iBody = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    const int iBody1 = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + 1;
    const float softSquared = soft * soft;

    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float4 q = make_float4(inBodies[iBody].qx, inBodies[iBody].qy, inBodies[iBody].qz, 0.0f);
    float4 q1 = make_float4(inBodies[iBody1].qx, inBodies[iBody1].qy, inBodies[iBody1].qz, 0.0f);

    // shared memory is too small to contains all the bodies
    // acumulate the acceleration in multiple passes
    for (int pass = 0; pass < nbPass; pass++) {
        const int startIdx = pass * sizePass;
        const int endIdx = min((pass + 1) * sizePass, nbBodies);
        unsigned int shIdx = 0;

        // load in shared memory
        shBodies[threadIdx.x] = inBodies[startIdx + threadIdx.x];
        shBodies[threadIdx.x + 768] = inBodies[startIdx + threadIdx.x + 768];
        __syncthreads();

        for (int jBody = startIdx; jBody < endIdx; jBody++) {
            float4 shBody = make_float4(shBodies[shIdx].qx, shBodies[shIdx].qy, shBodies[shIdx].qz, shBodies[shIdx].m);

            float rijx = shBody.x - q.x;
            float rijy = shBody.y - q.y;
            float rijz = shBody.z - q.z;

            float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared;
            float revSqrt = rsqrtf(rijSquared);
            float rsqrt3 = revSqrt * revSqrt * revSqrt;
            float ai = G * shBody.w * rsqrt3;

            a.x += ai * rijx;
            a.y += ai * rijy;
            a.z += ai * rijz;

            rijx = shBody.x - q1.x;
            rijy = shBody.y - q1.y;
            rijz = shBody.z - q1.z;

            rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared;
            revSqrt = rsqrtf(rijSquared);
            rsqrt3 = revSqrt * revSqrt * revSqrt;
            ai = G * shBody.w * rsqrt3;

            a1.x += ai * rijx;
            a1.y += ai * rijy;
            a1.z += ai * rijz;

            shIdx++;
        }

        __syncthreads();
    }

    outAccelerations[iBody].ax = a.x;
    outAccelerations[iBody].ay = a.y;
    outAccelerations[iBody].az = a.z;
    outAccelerations[iBody1].ax = a1.x;
    outAccelerations[iBody1].ay = a1.y;
    outAccelerations[iBody1].az = a1.z;
}

SimulationNBodyCudaOptim2::SimulationNBodyCudaOptim2(const unsigned long nBodies, const std::string &scheme,
                                                     const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = (20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN()) +
                        (9.0f * (float)this->getBodies().getN());
    this->accelerations.resize(this->getBodies().getN());
    this->packedBodies.resize(this->getBodies().getN());

    CUDA_CHECK(cudaMalloc(&cudaAccelerations, this->getBodies().getN() * sizeof(accAoS_t<float>)));
    CUDA_CHECK(cudaMalloc(&cudaBodies, this->getBodies().getN() * sizeof(cudaPackedAoS_t<float>)));
}

void SimulationNBodyCudaOptim2::initIteration()
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

void SimulationNBodyCudaOptim2::computeBodiesAcceleration()
{
    dim3 block(768);
    int nbBlocks = (this->getBodies().getN() + block.x - 1) / block.x;
    nbBlocks = (nbBlocks + 1) / 2;
    dim3 grid(nbBlocks);

    CUDA_CHECK(cudaMemcpy(this->cudaBodies, this->packedBodies.data(),
                          this->getBodies().getN() * sizeof(cudaPackedAoS_t<float>), cudaMemcpyHostToDevice));

    kernel_cuda_optim2<<<grid, block>>>(this->cudaBodies, this->cudaAccelerations, this->getBodies().getN(), this->soft,
                                        this->G, nbBlocks * block.x);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(this->accelerations.data(), this->cudaAccelerations,
                          this->getBodies().getN() * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost));
}

void SimulationNBodyCudaOptim2::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

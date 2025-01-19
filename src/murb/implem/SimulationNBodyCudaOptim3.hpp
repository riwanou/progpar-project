#ifndef SIMULATION_N_BODY_SIMD_CUDA_OPTIM3_HPP_
#define SIMULATION_N_BODY_SIMD_CUDA_OPTIM3_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCudaOptim3 : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    accAoS_t<float>* cudaAccelerations;
    float* cuda_qx;
    float* cuda_qy;
    float* cuda_qz;
    float* cuda_m;

  public:
    SimulationNBodyCudaOptim3(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                        const unsigned long randInit = 0);
    virtual ~SimulationNBodyCudaOptim3() = default;
    virtual void computeOneIteration();

  protected:
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_CUDA_OPTIM2_HPP_ */


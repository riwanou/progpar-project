#ifndef SIMULATION_N_BODY_SIMD_CUDA_OPTIM1_HPP_
#define SIMULATION_N_BODY_SIMD_CUDA_OPTIM1_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCudaOptim1 : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    accAoS_t<float>* cudaAccelerations;
    dataAoS_t<float>* cudaBodies;

  public:
    SimulationNBodyCudaOptim1(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                        const unsigned long randInit = 0);
    virtual ~SimulationNBodyCudaOptim1() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_CUDA_OPTIM1_HPP_ */


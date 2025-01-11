#ifndef SIMULATION_N_BODY_SIMD_CUDA_OPTIM2_HPP_
#define SIMULATION_N_BODY_SIMD_CUDA_OPTIM2_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

template <typename T>
struct cudaPackedAoS_t {
  T qx; /* position x. */
  T qy; /* position y. */
  T qz; /* position z. */
  T m; /* mass. */
};

class SimulationNBodyCudaOptim2 : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    std::vector<cudaPackedAoS_t<float>> packedBodies;
    accAoS_t<float>* cudaAccelerations;
    cudaPackedAoS_t<float>* cudaBodies;

  public:
    SimulationNBodyCudaOptim2(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                        const unsigned long randInit = 0);
    virtual ~SimulationNBodyCudaOptim2() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_CUDA_OPTIM2_HPP_ */


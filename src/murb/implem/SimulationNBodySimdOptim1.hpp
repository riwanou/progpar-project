#ifndef SIMULATION_N_BODY_SIMD_OPTIM1_HPP_
#define SIMULATION_N_BODY_SIMD_OPTIM1_HPP_

#include <string>

#include "core/Bodies.hpp"
#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodySimdOptim1 : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodySimdOptim1(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                        const unsigned long randInit = 0);
    virtual ~SimulationNBodySimdOptim1() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_OPTIM1_HPP_ */


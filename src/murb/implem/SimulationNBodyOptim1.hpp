#ifndef SIMULATION_N_BODY_NAIVE_OPTIM1_HPP_
#define SIMULATION_N_BODY_NAIVE_OPTIM1_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyOptim1 : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyOptim1(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOptim1() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_OPTIM1_HPP_ */

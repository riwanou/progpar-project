#ifndef SIMULATION_N_BODY_NAIVE_APPROX_HPP_
#define SIMULATION_N_BODY_NAIVE_APPROX_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyOptim1Approx : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyOptim1Approx(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOptim1Approx() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_APPROX_HPP_ */

#ifndef SIMULATION_N_BODY_ARRAY_HPP_
#define SIMULATION_N_BODY_ARRAY_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyArray : public SimulationNBodyInterface {
  protected:
    // std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    struct accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyArray(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyArray() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_ARRAY_HPP_ */

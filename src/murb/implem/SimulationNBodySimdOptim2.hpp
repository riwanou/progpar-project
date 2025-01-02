#ifndef SIMULATION_N_BODY_SIMD_OPTIM2_HPP_
#define SIMULATION_N_BODY_SIMD_OPTIM2_HPP_

#include <string>

#include "core/Bodies.hpp"
#include "core/SimulationNBodyInterface.hpp"

/**
 * In this optimization, we assume that simd register can contains multiple of 4 floats.
 */

 struct packedAoS_t {
    float qx; /* position x. */
    float qy; /* position y. */
    float qz; /* position z. */
    float m; /* mass. */
 };

class SimulationNBodySimdOptim2 : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */
    std::vector<packedAoS_t> packed_bodies;

  public:
    SimulationNBodySimdOptim2(const unsigned long nBodies, const std::string &scheme = "galaxy",
                              const float soft = 0.035f, const unsigned long randInit = 0);
    virtual ~SimulationNBodySimdOptim2() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_OPTIM2_HPP_ */

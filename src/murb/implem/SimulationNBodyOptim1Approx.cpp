#include <string>
#include <cmath>

#include "SimulationNBodyOptim1Approx.hpp"

SimulationNBodyOptim1Approx::SimulationNBodyOptim1Approx(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    const float N = this->getBodies().getN();
    this->flopsPerIte = 29.f * (N * (N + 1) / 2);
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyOptim1Approx::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyOptim1Approx::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    // compute e²
    const float softSquared = this->soft * this->soft;

    // flops = (n * (n + 1) / 2) * 29
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = 29 per iteration
        for (unsigned long jBody = iBody + 1; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² + e² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared; // 6 flops
            // compute the inverse distance between the bodies: 1 / (|| rij ||² + e²)^{3/2}            
            const float rsqrt = 1 / sqrt(rijSquared); // 2 flops
            const float rsqrt3 = rsqrt * rsqrt * rsqrt; // 2 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / distance
            const float ai = this->G * d[jBody].m * rsqrt3; // 2 flops
            // compute the acceleration value between body i and body j: || aj || = G.mj / distance
            const float aj = this->G * d[iBody].m * rsqrt3; // 2 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations[iBody].ax += ai * rijx; // 2 flops
            this->accelerations[iBody].ay += ai * rijy; // 2 flops
            this->accelerations[iBody].az += ai * rijz; // 2 flops
            // add the acceleration value into the acceleration vector: aj += || aj ||-.rij
            this->accelerations[jBody].ax += aj * -rijx; // 2 flops
            this->accelerations[jBody].ay += aj * -rijy; // 2 flops
            this->accelerations[jBody].az += aj * -rijz; // 2 flops
        }
    }
}

void SimulationNBodyOptim1Approx::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

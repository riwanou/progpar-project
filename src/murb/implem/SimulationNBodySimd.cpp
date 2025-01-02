#include "core/Bodies.hpp"
#include "mipp.h"
#include <cmath>
#include <string>

#include "SimulationNBodySimd.hpp"

SimulationNBodySimd::SimulationNBodySimd(const unsigned long nBodies, const std::string &scheme, const float soft,
                                         const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 23.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

void SimulationNBodySimd::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax[iBody] = 0.f;
        this->accelerations.ay[iBody] = 0.f;
        this->accelerations.az[iBody] = 0.f;
    }
}

void SimulationNBodySimd::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    mipp::Reg<float> r_qx_j, r_qy_j, r_qz_j, r_m_j;
    mipp::Reg<float> r_rijx, r_rijy, r_rijz, r_rijSquared;
    mipp::Reg<float> r_softFactor, r_ai;
    mipp::Reg<float> r_acc_x, r_acc_y, r_acc_z;
    // compute e²
    const float softSquared = std::pow(this->soft, 2);        // 1 flops
    mipp::Reg<float> r_softSquared = mipp::set1(softSquared); // 1 flops

    // tail loop
    size_t simd_loop_size = (this->getBodies().getN() / mipp::N<float>()) * mipp::N<float>();

    // simd
    // flops = n² * 23
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        float qx_i = d.qx[iBody];
        float qy_i = d.qy[iBody];
        float qz_i = d.qz[iBody];

        // simd
        // flops = n * 23
        for (unsigned long jBody = 0; jBody < simd_loop_size; jBody += mipp::N<float>()) {
            r_qx_j.load(&d.qx[jBody]);
            r_qy_j.load(&d.qy[jBody]);
            r_qz_j.load(&d.qz[jBody]);
            r_m_j.load(&d.m[jBody]);

            r_rijx = r_qx_j - qx_i; // 1 flop
            r_rijy = r_qy_j - qy_i; // 1 flop
            r_rijz = r_qz_j - qz_i; // 1 flop

            // compute the || rij ||² distance between body i and body j
            r_rijSquared = r_rijx * r_rijx + r_rijy * r_rijy + r_rijz * r_rijz; // 5 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            r_softFactor = (r_rijSquared + r_softSquared) * mipp::sqrt(r_rijSquared + r_softSquared); // 3 flops
            r_ai = (r_m_j * this->G) / r_softFactor;                                                  // 3 flops

            r_acc_x = r_ai * r_rijx;
            r_acc_y = r_ai * r_rijy;
            r_acc_z = r_ai * r_rijz;

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations.ax[iBody] += mipp::sum(r_ai * r_rijx); // 3
            this->accelerations.ay[iBody] += mipp::sum(r_ai * r_rijy); // 3
            this->accelerations.az[iBody] += mipp::sum(r_ai * r_rijz); // 3
        }

        // remaining, elements in the j-array don't fit in a simd register
        // flops = (remaining n) * 20
        for (unsigned long jBody = simd_loop_size; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d.qx[jBody] - qx_i; // 1 flop
            const float rijy = d.qy[jBody] - qy_i; // 1 flop
            const float rijz = d.qz[jBody] - qz_i; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float softFactor = (rijSquared + softSquared) * sqrt(rijSquared + softSquared); // 3 flops
            const float ai = this->G * d.m[jBody] / softFactor;                                   // 3 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            this->accelerations.ay[iBody] += ai * rijy; // 2 flops
            this->accelerations.az[iBody] += ai * rijz; // 2 flops
        }
    }
}

void SimulationNBodySimd::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

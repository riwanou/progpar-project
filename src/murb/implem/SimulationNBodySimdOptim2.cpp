#include "core/Bodies.hpp"
#include "mipp.h"
#include <cmath>
#include <string>

#include "SimulationNBodySimdOptim2.hpp"

SimulationNBodySimdOptim2::SimulationNBodySimdOptim2(const unsigned long nBodies, const std::string &scheme,
                                                     const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 23.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
    this->packed_bodies.resize(this->getBodies().getN());
}

void SimulationNBodySimdOptim2::initIteration()
{
    std::vector<dataAoS_t<float>> bodies = this->getBodies().getDataAoS();

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax[iBody] = 0.f;
        this->accelerations.ay[iBody] = 0.f;
        this->accelerations.az[iBody] = 0.f;

        dataAoS_t<float> body = bodies[iBody];
        this->packed_bodies[iBody].qx = body.qx;
        this->packed_bodies[iBody].qy = body.qy;
        this->packed_bodies[iBody].qz = body.qz;
        this->packed_bodies[iBody].m = body.m;
    }
}

void SimulationNBodySimdOptim2::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    mipp::Reg<float> r_rijx, r_rijy, r_rijz, r_rijSquared;
    mipp::Reg<float> r_rsqrt, r_softFactor, r_ai;
    mipp::Reg<float> r_ax, r_ay, r_az, r_acc_x, r_acc_y, r_acc_z;
    mipp::Reg<float> r_qx_i, r_qy_i, r_qz_i, r_m_i;
    // AoS access pattern
    mipp::Reg<float> r_packed_j, r_qx_j, r_qy_j, r_qz_j, r_m_j;
    // compute e²
    const float softSquared = std::pow(this->soft, 2);        // 1 flops
    mipp::Reg<float> r_softSquared = mipp::set1(softSquared); // 1 flops

    // tail loop
    size_t simd_loop_size = (this->getBodies().getN() / mipp::N<float>()) * mipp::N<float>();

    // remaining
    // flops = n² * 23
    for (unsigned long iBody = 0; iBody < simd_loop_size; iBody += mipp::N<float>()) {
        r_qx_i.load(&d.qx[iBody]);
        r_qy_i.load(&d.qy[iBody]);
        r_qz_i.load(&d.qz[iBody]);

        r_ax.load(&this->accelerations.ax[iBody]);
        r_ay.load(&this->accelerations.ay[iBody]);
        r_az.load(&this->accelerations.az[iBody]);

        // simd
        // flops = n * 23
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody += 1) {
            r_packed_j.load((const float *)(&this->packed_bodies[jBody]));
            r_qx_j = r_packed_j.get(0);
            r_qy_j = r_packed_j.get(1);
            r_qz_j = r_packed_j.get(2);
            r_m_j = r_packed_j.get(3);

            r_rijx = r_qx_j - r_qx_i; // 1 flop
            r_rijy = r_qy_j - r_qy_i; // 1 flop
            r_rijz = r_qz_j - r_qz_i; // 1 flop

            // compute the || rij ||² + e² distance between body i and body j
            r_rijSquared = (r_rijx * r_rijx + r_rijy * r_rijy + r_rijz * r_rijz) + r_softSquared; // 6 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            // a / b^(3/2) equivalent to: a * (1 / sqrt(b) * (1 / sqrt(b)) * (1 / sqrt(b))
            r_rsqrt = mipp::rsqrt(r_rijSquared);                      // 1 flop
            r_ai = (r_m_j * this->G) * (r_rsqrt * r_rsqrt * r_rsqrt); // 4 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            r_ax += r_ai * r_rijx;
            r_ay += r_ai * r_rijy;
            r_az += r_ai * r_rijz;
        }

        // store
        r_ax.store(&this->accelerations.ax[iBody]);
        r_ay.store(&this->accelerations.ay[iBody]);
        r_az.store(&this->accelerations.az[iBody]);
    }

    // remaining
    // flops = n² * 23
    for (unsigned long iBody = simd_loop_size; iBody < this->getBodies().getN(); iBody++) {
        float qx_i = d.qx[iBody];
        float qy_i = d.qy[iBody];
        float qz_i = d.qz[iBody];
        float ax = this->accelerations.ax[iBody];
        float ay = this->accelerations.ay[iBody];
        float az = this->accelerations.az[iBody];

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

            // compute the || rij ||² + e² distance between body i and body j
            r_rijSquared = (r_rijx * r_rijx + r_rijy * r_rijy + r_rijz * r_rijz) + r_softSquared; // 6 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            // a / b^(3/2) equivalent to: a * (1 / sqrt(b) * (1 / sqrt(b)) * (1 / sqrt(b))
            r_rsqrt = mipp::rsqrt(r_rijSquared);                      // 1 flop
            r_ai = (r_m_j * this->G) * (r_rsqrt * r_rsqrt * r_rsqrt); // 4 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            ax += mipp::sum(r_ai * r_rijx); // 3
            ay += mipp::sum(r_ai * r_rijy); // 3
            az += mipp::sum(r_ai * r_rijz); // 3
        }

        // remaining, elements in the j-array don't fit in a simd register
        // flops = (remaining n) * 21
        for (unsigned long jBody = simd_loop_size; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d.qx[jBody] - qx_i; // 1 flop
            const float rijy = d.qy[jBody] - qy_i; // 1 flop
            const float rijz = d.qz[jBody] - qz_i; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared; // 6 flops
            // compute the inverse distance between the bodies: 1 / (|| rij ||² + e²)^{3/2}
            const float rsqrt = 1 / sqrt(rijSquared);   // 2 flops
            const float rsqrt3 = rsqrt * rsqrt * rsqrt; // 2 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / distance
            const float ai = this->G * d.m[jBody] * rsqrt3; // 2 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops
        }

        // store
        this->accelerations.ax[iBody] = ax;
        this->accelerations.ay[iBody] = ay;
        this->accelerations.az[iBody] = az;
    }
}

void SimulationNBodySimdOptim2::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

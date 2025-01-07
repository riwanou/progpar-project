#include "core/Bodies.hpp"
#include "mipp.h"
#include <cmath>
#include <string>

#include "SimulationNBodySimdOptim2.hpp"

SimulationNBodySimdOptim2::SimulationNBodySimdOptim2(const unsigned long nBodies, const std::string &scheme,
                                                     const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
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
    std::array<uint32_t, mipp::N<float>()> mask;
    mask.fill(0);
    mipp::Reg<float> r_shuf_x = mipp::cmask<float>(mask.data());
    mask.fill(1);
    mipp::Reg<float> r_shuf_y = mipp::cmask<float>(mask.data());
    mask.fill(2);
    mipp::Reg<float> r_shuf_z = mipp::cmask<float>(mask.data());
    mask.fill(3);
    mipp::Reg<float> r_shuf_m = mipp::cmask<float>(mask.data());
    // compute e²
    const float softSquared = std::pow(this->soft, 2);        // 1 flops
    mipp::Reg<float> r_softSquared = mipp::set1(softSquared); // 1 flops

    // tail loop
    size_t simd_loop_size = (this->getBodies().getN() / mipp::N<float>()) * mipp::N<float>();

    // remaining
    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < simd_loop_size; iBody += mipp::N<float>()) {
        r_qx_i.load(&d.qx[iBody]);
        r_qy_i.load(&d.qy[iBody]);
        r_qz_i.load(&d.qz[iBody]);

        r_ax.load(&this->accelerations.ax[iBody]);
        r_ay.load(&this->accelerations.ay[iBody]);
        r_az.load(&this->accelerations.az[iBody]);

        // simd
        // flops = n * 20
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody += 1) {
            r_packed_j.load((const float *)(&this->packed_bodies[jBody]));
            r_qx_j = mipp::shuff(r_packed_j, r_shuf_x);
            r_qy_j = mipp::shuff(r_packed_j, r_shuf_y);
            r_qz_j = mipp::shuff(r_packed_j, r_shuf_z);
            r_m_j = mipp::shuff(r_packed_j, r_shuf_m);

            r_rijx = r_qx_j - r_qx_i; // 1 flop
            r_rijy = r_qy_j - r_qy_i; // 1 flop
            r_rijz = r_qz_j - r_qz_i; // 1 flop

            // compute the || rij ||² + e² distance between body i and body j
            r_rijSquared = (r_rijx * r_rijx + r_rijy * r_rijy + r_rijz * r_rijz) + r_softSquared; // 6 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            // a / b^(3/2) equivalent to: a * (1 / sqrt(b) * (1 / sqrt(b)) * (1 / sqrt(b))
            r_rsqrt = mipp::rsqrt_prec(r_rijSquared);                 // 1 flop
            r_ai = (r_m_j * this->G) * (r_rsqrt * r_rsqrt * r_rsqrt); // 4 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            r_ax = mipp::fmadd(r_ai, r_rijx, r_ax); // 2 flops
            r_ay = mipp::fmadd(r_ai, r_rijy, r_ay); // 2 flops
            r_az = mipp::fmadd(r_ai, r_rijz, r_az); // 2 flops
        }

        // store
        r_ax.store(&this->accelerations.ax[iBody]);
        r_ay.store(&this->accelerations.ay[iBody]);
        r_az.store(&this->accelerations.az[iBody]);
    }

    // remaining
    // flops = n² * 20
    for (unsigned long iBody = simd_loop_size; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d.qx[jBody] - d.qx[iBody]; // 1 flop
            const float rijy = d.qy[jBody] - d.qy[iBody]; // 1 flop
            const float rijz = d.qz[jBody] - d.qz[iBody]; // 1 flop

            // compute the || rij ||² + e² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared; // 6 flops
            // compute the inverse distance between the bodies: 1 / (|| rij ||² + e²)^{3/2}
            const float rsqrt = 1 / sqrt(rijSquared);   // 2 flops
            const float rsqrt3 = rsqrt * rsqrt * rsqrt; // 2 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / distance
            const float ai = this->G * d.m[jBody] * rsqrt3; // 2 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            this->accelerations.ay[iBody] += ai * rijy; // 2 flops
            this->accelerations.az[iBody] += ai * rijz; // 2 flops
        }
    }
}

void SimulationNBodySimdOptim2::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

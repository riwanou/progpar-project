#include <catch.hpp>
#include <string>

#include "SimulationNBodyNaive.hpp"
#include "SimulationNBodyCudaOptim1.hpp"

void test_nbody_cuda_optim1(const size_t n, const float soft, const float dt, const size_t nIte, const std::string &scheme,
                     const float eps)
{
    SimulationNBodyNaive simuRef(n, scheme, soft);
    simuRef.setDt(dt);

    SimulationNBodyCudaOptim1 simuTest(n, scheme, soft);
    simuTest.setDt(dt);

    const float *xRef = simuRef.getBodies().getDataSoA().qx.data();
    const float *yRef = simuRef.getBodies().getDataSoA().qy.data();
    const float *zRef = simuRef.getBodies().getDataSoA().qz.data();

    const float *xTest = simuTest.getBodies().getDataSoA().qx.data();
    const float *yTest = simuTest.getBodies().getDataSoA().qy.data();
    const float *zTest = simuTest.getBodies().getDataSoA().qz.data();

    float e = 0; // espilon
    for (size_t i = 0; i < nIte + 1; i++) {
        if (i > 0) {
            simuRef.computeOneIteration();
            simuTest.computeOneIteration();
            e = eps;
        }

        for (size_t b = 0; b < simuRef.getBodies().getN(); b++) {
            REQUIRE_THAT(xRef[b], Catch::Matchers::WithinRel(xTest[b], e));
            REQUIRE_THAT(yRef[b], Catch::Matchers::WithinRel(yTest[b], e));
            REQUIRE_THAT(zRef[b], Catch::Matchers::WithinRel(zTest[b], e));
        }
    }
}

TEST_CASE("n-body - Cuda optim1", "[cuda+optim1]")
{
    SECTION("fp32 - n=13 - i=1 - random") { test_nbody_cuda_optim1(13, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=13 - i=100 - random") { test_nbody_cuda_optim1(13, 2e+08, 3600, 100, "random", 5e-3); }
    SECTION("fp32 - n=16 - i=1 - random") { test_nbody_cuda_optim1(16, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=128 - i=1 - random") { test_nbody_cuda_optim1(128, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=2048 - i=1 - random") { test_nbody_cuda_optim1(2048, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=2049 - i=3 - random") { test_nbody_cuda_optim1(2049, 2e+08, 3600, 3, "random", 1e-3); }

    SECTION("fp32 - n=13 - i=1 - galaxy") { test_nbody_cuda_optim1(13, 2e+08, 3600, 1, "galaxy", 1e-1); }
    SECTION("fp32 - n=13 - i=30 - galaxy") { test_nbody_cuda_optim1(13, 2e+08, 3600, 30, "galaxy", 1e-1); }
    SECTION("fp32 - n=16 - i=1 - galaxy") { test_nbody_cuda_optim1(16, 2e+08, 3600, 1, "galaxy", 1e-2); }
    SECTION("fp32 - n=128 - i=1 - galaxy") { test_nbody_cuda_optim1(128, 2e+08, 3600, 1, "galaxy", 1e-2); }
    SECTION("fp32 - n=2048 - i=4 - galaxy") { test_nbody_cuda_optim1(2048, 2e+08, 3600, 4, "galaxy", 1e-1); }
    SECTION("fp32 - n=2049 - i=3 - galaxy") { test_nbody_cuda_optim1(2049, 2e+08, 3600, 3, "galaxy", 1e-1); }

}


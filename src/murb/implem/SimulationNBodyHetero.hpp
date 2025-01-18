#ifndef SIMULATION_N_BODY_HETERO_HPP_
#define SIMULATION_N_BODY_HETERO_HPP_

#include "core/SimulationNBodyInterface.hpp"

#include <CL/cl.h>

class SimulationNBodyHetero : public SimulationNBodyInterface {
protected:
	/*!< Array of body acceleration structures. */
	accSoA_t<float> accelerations;

	cl_mem ax_buffer;
	cl_mem ay_buffer;
	cl_mem az_buffer;
	cl_mem data_buffer;

private:
	cl_kernel kernel;
	cl_context context;
	cl_program program;
	cl_command_queue command_queue;
	
	size_t global_work_size;
	size_t local_work_size;

public:
	SimulationNBodyHetero(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f, const unsigned long randInit = 0);
	virtual ~SimulationNBodyHetero() = default;
	virtual void computeOneIteration();
protected:
	void computeBodiesAccelerationGPU();
	void computeBodiesAccelerationCPU();
};
#endif /* SIMULATION_N_BODY_HETERO_HPP_ */

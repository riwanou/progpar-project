#ifndef SIMULATION_N_BODY_OCL_naive_HPP_
#define SIMULATION_N_BODY_OCL_naive_HPP_

#include "core/SimulationNBodyInterface.hpp"

#include <CL/cl.h>

class SimulationNBodyOCLNaive : public SimulationNBodyInterface {
protected:
	/*!< Array of body acceleration structures. */
	std::vector<accAoS_t<float>> accelerations;
	cl_mem accelerations_buffer;
	cl_mem data_buffer;

private:
	cl_kernel kernel;
	cl_context context;
	cl_program program;
	cl_command_queue command_queue;
	
	size_t global_work_size;
	size_t local_work_size;

public:
	SimulationNBodyOCLNaive(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f, const unsigned long randInit = 0);
	virtual ~SimulationNBodyOCLNaive() = default;
	virtual void computeOneIteration();
protected:
	void computeBodiesAcceleration();
};
#endif /* SIMULATION_N_BODY_OCL_NAIVE_HPP_ */
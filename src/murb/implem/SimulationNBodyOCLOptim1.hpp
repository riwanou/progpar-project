#ifndef SIMULATION_N_BODY_OCL_OPTIM1_HPP_
#define SIMULATION_N_BODY_OCL_OPTIM1_HPP_

#include "core/SimulationNBodyInterface.hpp"

#include <CL/cl.h>

class SimulationNBodyOCL : public SimulationNBodyInterface {
protected:
	/*!< Array of body acceleration structures. */
	std::vector<accAoS_t<float>> accelerations;

	cl_mem accelerations_buffer;
	cl_mem data_buffer;

private:
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	
	size_t global_work_size;
    size_t local_work_size;

public:
	SimulationNBodyOCL(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f, const unsigned long randInit = 0);
	virtual ~SimulationNBodyOCL();
	virtual void computeOneIteration();

protected:
	void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OCL_OPTIM1_HPP_ */
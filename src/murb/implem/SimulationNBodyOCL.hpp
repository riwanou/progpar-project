#ifndef SIMULATION_N_BODY_OCL_HPP_
#define SIMULATION_N_BODY_OCL_HPP_

#include "core/SimulationNBodyInterface.hpp"

#include <CL/cl.h>

class SimulationNBodyOCL : public SimulationNBodyInterface {
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

	void create_buffers();
	void get_kernel_code();

public:
	SimulationNBodyOCL(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f, const unsigned long randInit = 0);
	virtual ~SimulationNBodyOCL();
	virtual void computeOneIteration();

protected:
	void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OCL_HPP_ */
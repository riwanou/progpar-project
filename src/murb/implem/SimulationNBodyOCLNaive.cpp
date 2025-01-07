#include <cmath>

#include <iostream>
#include <stdexcept>

#include "SimulationNBodyOCLNaive.hpp"

#define KERNEL_PATH "../src/murb/implem/kernel.cl"
#define KERNEL_FUNC "compute_bodies_acceleration"

#define CHECK_CL_ERR(err, msg) if (err != CL_SUCCESS) { \
	std::cerr << msg << " (Error Code: " << err << ")\n"; \
	throw std::runtime_error(msg); \
}

#define PLUS_GRAND_DIVISEUR(n) ({                      \
    int diviseur = 0;                                  \
    for (int i = ((n) > 256 ? 256 : (n) - 1); i > 0; i--) { \
        if ((n) % i == 0) {                            \
            diviseur = i;                              \
            break;                                     \
        }                                              \
    }                                                  \
    diviseur;                                          \
})

SimulationNBodyOCLNaive::SimulationNBodyOCLNaive(const unsigned long nBodies, const std::string &scheme, const float soft, const unsigned long randInit)
	: SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
	this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
	this->accelerations.resize(this->getBodies().getN());

	// OCL INITIALISATION
	cl_uint num_platforms;
	cl_int err;
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_CL_ERR(err, "Failed to get number of platforms");

	cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
	if (!platforms) {
		throw std::runtime_error("Failed to allocate memory for platforms");
	}

	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_CL_ERR(err, "Failed to get platform IDs");
	
	cl_uint num_devices;
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	CHECK_CL_ERR(err, "Failed to get number of devices");
	
	cl_device_id *devices_list = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
	if (!devices_list) {
		free(platforms);
		throw std::runtime_error("Failed to allocate memory for devices list");
	}
	
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices_list, NULL);
	CHECK_CL_ERR(err, "Failed to get device IDs");

	this->context = clCreateContext(NULL, num_devices, devices_list, NULL, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create OpenCL context");

	this->command_queue = clCreateCommandQueue(this->context, devices_list[0], 0, NULL);
	CHECK_CL_ERR(err, "Failed to create OpenCL command queue");

	free(platforms);

	FILE *kernels_file = fopen(KERNEL_PATH, "r");
	if (!kernels_file) {
		free(devices_list);
		throw std::runtime_error("Failed to open kernel source file");
	}
	fseek(kernels_file, 0, SEEK_END);
	size_t file_size = ftell(kernels_file);
	fseek(kernels_file, 0, SEEK_SET);
	char *kernels_source = (char *) malloc((file_size + 1) * sizeof(char));
	if (!kernels_source) {
		fclose(kernels_file);
		free(devices_list);
		throw std::runtime_error("Failed to allocate memory for kernel source");
	}

	fread(kernels_source, sizeof(char), file_size, kernels_file);
	kernels_source[file_size] = '\0';
	fclose(kernels_file);

	this->program = clCreateProgramWithSource(context, 1, (const char **) &kernels_source, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create OpenCL program");

	clBuildProgram(program, 1, &(devices_list[0]), NULL, NULL, NULL);


	this->kernel = clCreateKernel(program, KERNEL_FUNC, NULL);
	CHECK_CL_ERR(err, "Failed to create OpenCL kernel");

	this->accelerations_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(accAoS_t<float>) * this->getBodies().getN(), NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create accelerations buffer");

	this->data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(dataAoS_t<float>) * this->getBodies().getN(), NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create data buffer");

	size_t max_work_group_size;
	err = clGetDeviceInfo(devices_list[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_CL_ERR(err, "Failed to get max work group size");


	this->global_work_size = this->getBodies().getN();
	this->local_work_size = PLUS_GRAND_DIVISEUR(this->getBodies().getN());

	free(kernels_source);
	free(devices_list);
}

void SimulationNBodyOCLNaive::computeBodiesAcceleration()
{
	const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

	clEnqueueWriteBuffer(this->command_queue, this->data_buffer, CL_TRUE, 0, sizeof(dataAoS_t<float>) * this->getBodies().getN(), d.data(), 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->accelerations_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->data_buffer);
	clSetKernelArg(kernel, 2, sizeof(float), &this->soft);
	clSetKernelArg(kernel, 3, sizeof(float), &this->G);

	clEnqueueNDRangeKernel(this->command_queue, kernel, 1, NULL, &this->global_work_size, &this->local_work_size, 0, NULL, NULL);
	clEnqueueReadBuffer(this->command_queue, this->accelerations_buffer, CL_TRUE, 0, sizeof(accAoS_t<float>) * this->getBodies().getN(), this->accelerations.data(), 0, NULL, NULL);
}

void SimulationNBodyOCLNaive::computeOneIteration()
{
	this->computeBodiesAcceleration();
	// time integration
	this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
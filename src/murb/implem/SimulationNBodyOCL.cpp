#include "SimulationNBodyOCL.hpp"
#include <iostream>
#include <stdexcept>

#define CHECK_CL_ERR(err, msg) if (err != CL_SUCCESS) { \
	std::cerr << msg << " (Error Code: " << err << ")\n"; \
	throw std::runtime_error(msg); \
}

SimulationNBodyOCL::SimulationNBodyOCL(const unsigned long nBodies, const std::string &scheme, const float soft, const unsigned long randInit) : SimulationNBodyInterface(nBodies, scheme, soft, randInit) 
{
	this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
	this->accelerations.resize(this->getBodies().getN());

	cl_int err;
	cl_uint num_platforms;
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_CL_ERR(err, "Failed to get number of platforms");

	cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
	if (!platforms) {
		throw std::runtime_error("Failed to allocate memory for platforms");
	}

	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_CL_ERR(err, "Failed to get platform IDs");

	cl_uint num_devices;
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	CHECK_CL_ERR(err, "Failed to get number of devices");

	cl_device_id *devices_list = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
	if (!devices_list) {
		free(platforms);
		throw std::runtime_error("Failed to allocate memory for devices list");
	}

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices_list, NULL);
	CHECK_CL_ERR(err, "Failed to get device IDs");

	this->context = clCreateContext(NULL, num_devices, devices_list, NULL, NULL, &err);
	CHECK_CL_ERR(err, "Failed to create OpenCL context");

	this->command_queue = clCreateCommandQueue(this->context, devices_list[0], 0, &err);
	CHECK_CL_ERR(err, "Failed to create OpenCL command queue");

	free(platforms);

	FILE *kernels_file = fopen("../src/murb/implem/kernel_compute_acceleration.cl", "r");
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

	this->program = clCreateProgramWithSource(context, 1, (const char **) &kernels_source, NULL, &err);
	CHECK_CL_ERR(err, "Failed to create OpenCL program");

	err = clBuildProgram(program, 1, &(devices_list[0]), NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(program, devices_list[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		char *build_log = (char *) malloc(log_size);
		if (build_log) {
			clGetProgramBuildInfo(program, devices_list[0],CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
			std::cerr << "OpenCL Program Build Error:\n" << build_log << "\n";
			free(build_log);
		} else {
			std::cerr << "Failed to allocate memory for build log\n";
		}

		free(kernels_source);
		free(devices_list);
		throw std::runtime_error("Failed to build OpenCL program");
	}

	this->kernel = clCreateKernel(program, "compute_acceleration", &err);
	CHECK_CL_ERR(err, "Failed to create OpenCL kernel");

	this->accelerations_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(accAoS_t<float>) * this->getBodies().getN(), NULL, &err);
	CHECK_CL_ERR(err, "Failed to create accelerations buffer");

	this->data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(dataAoS_t<float>) * this->getBodies().getN(), NULL, &err);
	CHECK_CL_ERR(err, "Failed to create data buffer");

	size_t max_work_group_size;
	err = clGetDeviceInfo(devices_list[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_CL_ERR(err, "Failed to get max work group size");

	this->global_work_size = this->getBodies().getN();
	this->local_work_size = (this->global_work_size < max_work_group_size) ? this->global_work_size : max_work_group_size;

	if (this->global_work_size % this->local_work_size != 0) {
		this->global_work_size = ((this->global_work_size + this->local_work_size - 1) / this->local_work_size) * this->local_work_size;
	}

	this->local_work_size = 256;

	free(kernels_source);
	free(devices_list);
}

SimulationNBodyOCL::~SimulationNBodyOCL() 
{
	clReleaseMemObject(this->accelerations_buffer);
	clReleaseMemObject(this->data_buffer);
	clReleaseKernel(this->kernel);
	clReleaseProgram(this->program);
	clReleaseCommandQueue(this->command_queue);
	clReleaseContext(this->context);
}

void SimulationNBodyOCL::computeBodiesAcceleration() 
{
	const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

	cl_int err = clEnqueueWriteBuffer(this->command_queue, this->data_buffer, CL_TRUE, 0, sizeof(dataAoS_t<float>) * this->getBodies().getN(), d.data(), 0, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to write to data buffer");

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->accelerations_buffer);
	CHECK_CL_ERR(err, "Failed to set kernel argument 0");

	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->data_buffer);
	CHECK_CL_ERR(err, "Failed to set kernel argument 1");

	err = clSetKernelArg(kernel, 2, sizeof(float), &this->soft);
	CHECK_CL_ERR(err, "Failed to set kernel argument 2");

	err = clSetKernelArg(kernel, 3, sizeof(float), &this->G);
	CHECK_CL_ERR(err, "Failed to set kernel argument 3");

	err = clEnqueueNDRangeKernel(this->command_queue, kernel, 1, NULL, &this->global_work_size, &this->local_work_size, 0, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to enqueue ND range kernel");

	err = clEnqueueReadBuffer(this->command_queue, this->accelerations_buffer, CL_TRUE, 0, sizeof(accAoS_t<float>) * this->getBodies().getN(), this->accelerations.data(), 0, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to read buffer");
}

void SimulationNBodyOCL::computeOneIteration() 
{
	this->computeBodiesAcceleration();
	this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
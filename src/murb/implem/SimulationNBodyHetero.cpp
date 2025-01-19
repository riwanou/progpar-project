#include <cmath>

#include "mipp.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "SimulationNBodyHetero.hpp"

#define KERNEL_PATH "../src/murb/implem/kernel.cl"
#define KERNEL_FUNC "compute_bodies_acceleration_hetero"

#define CHECK_CL_ERR(err, msg) if (err != CL_SUCCESS) { \
	std::cerr << msg << " (Error Code: " << err << ")\n"; \
	throw std::runtime_error(msg); \
}

SimulationNBodyHetero::SimulationNBodyHetero(const unsigned long nBodies, const std::string &scheme, const float soft, const unsigned long randInit)
	: SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
	const float N = this->getBodies().getN();
  const float simdFlopsPerIte = (20.f * N * N) + (6.0f * N);
  const float gpuFlopsPerIte = (20.f * N * N) + N;

	const float cpu_percent = 0.1f;
	this->flopsPerIte = cpu_percent * simdFlopsPerIte + (1.0f - cpu_percent) * gpuFlopsPerIte;


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

	size_t max_work_group_size;
	err = clGetDeviceInfo(devices_list[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_CL_ERR(err, "Failed to get max work group size");

	this->local_work_size = 64;
    this->global_work_size = this->getBodies().getN() * (1.0f - cpu_percent);
    this->global_work_size -= this->global_work_size % local_work_size;


	this->ax_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * this->global_work_size, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create accelerations buffer");
	this->ay_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * this->global_work_size, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create accelerations buffer");
	this->az_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * this->global_work_size, NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create accelerations buffer");

	this->data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(dataAoS_t<float>) * this->getBodies().getN(), NULL, NULL);
	CHECK_CL_ERR(err, "Failed to create data buffer");

	free(kernels_source);
	free(devices_list);
}

void SimulationNBodyHetero::computeBodiesAccelerationGPU()
{
	const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

	clEnqueueWriteBuffer(this->command_queue, this->data_buffer, CL_TRUE, 0, sizeof(dataAoS_t<float>) * this->getBodies().getN(), d.data(), 0, NULL, NULL);

	int N = this->getBodies().getN();
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->ax_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->ay_buffer);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->az_buffer);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->data_buffer);
	clSetKernelArg(kernel, 4, sizeof(float), &this->soft);
	clSetKernelArg(kernel, 5, sizeof(float), &this->G);
	clSetKernelArg(kernel, 6, sizeof(int), &N);

	clEnqueueNDRangeKernel(this->command_queue, kernel, 1, NULL, &this->global_work_size, &this->local_work_size, 0, NULL, NULL);
}

void SimulationNBodyHetero::computeBodiesAccelerationCPU()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    mipp::Reg<float> r_qx_j, r_qy_j, r_qz_j, r_m_j;
    mipp::Reg<float> r_rijx, r_rijy, r_rijz, r_rijSquared;
    mipp::Reg<float> r_rsqrt, r_softFactor, r_ai;
    mipp::Reg<float> r_ax, r_ay, r_az; 
    // compute e²
    const float softSquared = this->soft * this->soft;
    mipp::Reg<float> r_softSquared = mipp::set1(softSquared);

    // tail loop
    size_t simd_loop_size = (this->getBodies().getN() / mipp::N<float>()) * mipp::N<float>();

	// printf("%ld\n", this->global_work_size);
    // simd
    // flops = n² * 19 + n * 9
	#pragma omp parallel \
				for schedule(dynamic, (this->getBodies().getN() - this->global_work_size) / 6) \
                firstprivate(d, r_qx_j, r_qy_j, r_qz_j, r_m_j, r_rsqrt, \
                             r_rijx, r_rijy, r_rijz, r_rijSquared, \
                             r_softFactor, r_ai, r_ax, r_ay, r_az, \
                             r_softSquared, simd_loop_size)

    for (unsigned long iBody = this->global_work_size; iBody < this->getBodies().getN(); iBody++) {
        float qx_i = d.qx[iBody];
        float qy_i = d.qy[iBody];
        float qz_i = d.qz[iBody];

        r_ax = 0.f;
        r_ay = 0.f;
        r_az = 0.f;

        // simd
        // flops = n * 19
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
            r_rsqrt = mipp::rsqrt_prec(r_rijSquared); // 1 flop
            r_ai = r_m_j * (r_rsqrt * r_rsqrt * r_rsqrt); // 3 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            r_ax += r_ai * r_rijx; // 2 flops
            r_ay += r_ai * r_rijy; // 2 flops
            r_az += r_ai * r_rijz; // 2 flops
        }

        this->accelerations.ax[iBody] = mipp::sum(r_ax) * this->G; // 2 flops
        this->accelerations.ay[iBody] = mipp::sum(r_ay) * this->G; // 2 flops
        this->accelerations.az[iBody] = mipp::sum(r_az) * this->G; // 2 flops

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
            this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            this->accelerations.ay[iBody] += ai * rijy; // 2 flops
            this->accelerations.az[iBody] += ai * rijz; // 2 flops
        }
    }
}

void SimulationNBodyHetero::computeOneIteration()
{
	this->computeBodiesAccelerationGPU();
	this->computeBodiesAccelerationCPU();
	clFinish(command_queue);
	clEnqueueReadBuffer(this->command_queue, this->ax_buffer, CL_TRUE, 0, sizeof(float) * this->global_work_size, this->accelerations.ax.data(), 0, NULL, NULL);
	clEnqueueReadBuffer(this->command_queue, this->ay_buffer, CL_TRUE, 0, sizeof(float) * this->global_work_size, this->accelerations.ay.data(), 0, NULL, NULL);
	clEnqueueReadBuffer(this->command_queue, this->az_buffer, CL_TRUE, 0, sizeof(float) * this->global_work_size, this->accelerations.az.data(), 0, NULL, NULL);
	// time integration
	this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

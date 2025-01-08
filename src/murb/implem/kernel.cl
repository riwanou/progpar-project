// Data structure
typedef struct {
    float ax; // Acceleration x
    float ay; // Acceleration y
    float az; // Acceleration z
} accAoS_t;

typedef struct dataAoS_t {
    float qx; /*!< Position x. */
    float qy; /*!< Position y. */
    float qz; /*!< Position z. */
    float vx; /*!< Velocity x. */
    float vy; /*!< Velocity y. */
    float vz; /*!< Velocity z. */
    float m;  /*!< Mass. */
    float r;  /*!< Radius. */
} dataAoS_t;

// Kernel
__kernel void compute_bodies_acceleration(
	__global accAoS_t* accelerations,
	__global const dataAoS_t* data, 
	const float soft,
	const float G) 
{
	int iBody = get_global_id(0);

	float ax = 0.0f;
	float ay = 0.0f;
	float az = 0.0f;

	float softSquared = soft * soft; 

	for (int jBody = 0; jBody < get_global_size(0); jBody++) {
		// const float rijx = data[jBody].qx - data[iBody].qx;
		// const float rijy = data[jBody].qy - data[iBody].qy;
		// const float rijz = data[jBody].qz - data[iBody].qz;

		// const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz + softSquared;
        // const float rsqrt = native_rsqrt(rijSquared);
        // const float rsqrt3 = rsqrt * rsqrt * rsqrt;
		// const float ai = G * data[jBody].m * rsqrt3;

		ax += data[jBody].qx;
		ay += ai * rijy;
		az += ai * rijz;
	}

	accelerations[iBody].ax = ax;
	accelerations[iBody].ay = ay;
	accelerations[iBody].az = az;
}
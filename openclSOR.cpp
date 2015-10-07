/*Parallel Successive over Relaxation for Laplace equation 
* Rajmohan Asokan
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
#include <iomanip>
#include <cmath>
#include <ctime>
#define M_PI 3.14159265358979323846

// global variables - double precision is not supported by this OpenCL Device
size_t const BLOCK_SIZE = 16;
int width = 800;
int height = 800;
int itmax = 1000;
float const omega = 1.97;
float const beta = (((1.0/width) / (1.0/height))*((1.0/width) / (1.0/height)));


// functions
void generategrid(float*, float*, const float, const float, const float, const float); // Generates the rectangular grid
void setBC(float*, const float*, const float*); // Set boundary conditions
void write_output(float*); // Write the solution
void errorFunc(cl_int, char*); // Error function


int main(){
	cl_int errVal;
	cl_uint numPlatforms;
	cl_platform_id* platformID;
	cl_context context_mm = NULL;

	//Finding the number of platforms
	errVal = clGetPlatformIDs(0, NULL, &numPlatforms);
	errorFunc(errVal, "clGetPlatformIDs");
	{
		std::cout<< "The number of available platforms are: "<< numPlatforms<< std::endl;
	}

	platformID = (cl_platform_id *)alloca(sizeof(cl_platform_id)*numPlatforms);

	errVal = clGetPlatformIDs(numPlatforms, platformID, NULL);
	errorFunc(errVal, "clGetPlatformIDs");
	
	
	size_t sizeProfile;
	errVal = clGetPlatformInfo(platformID[0], CL_PLATFORM_VENDOR, 0, NULL, &sizeProfile);
	errorFunc(errVal, "clGetPlatformInfo");	

	cl_uint numDevices;
	errVal = clGetDeviceIDs(platformID[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	errorFunc(errVal, "clGetDeviceIDs");
	

	cl_device_id *deviceID;
	deviceID = (cl_device_id *)alloca(sizeof(cl_device_id)*numDevices);

	errVal = clGetDeviceIDs(platformID[0], CL_DEVICE_TYPE_GPU, numDevices, &deviceID[0], NULL);
	errorFunc(errVal, "clGetDeviceIDs");
	
	size_t sizeinfoDevice;
	size_t infoDevice;

	errVal = clGetDeviceInfo(deviceID[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &infoDevice, &sizeinfoDevice);
	errorFunc(errVal, "clGetDeviceInfo");
	{
		std::cout<< "Number of Compute Devices: "<< infoDevice<< std::endl;
	}

	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformID[0], 0};

	context_mm = clCreateContext(properties, numDevices, &deviceID[0], NULL, NULL, &errVal);
	errorFunc(errVal, "clCreateContext");
	{
		std::cout<< "CL Context successfully created"<< std::endl;
	}

	cl_command_queue queue = clCreateCommandQueue(context_mm, deviceID[0], 0, &errVal);
	errorFunc(errVal, "clCreateCommandQueue");

  //host variables
  float *x, *y;        // grid x and y
  float Xmin = 0.0, Xmax = 1.0,
        Ymin = 0.0, Ymax = 1.0;    // grid coordinates bounds
  float *solution;

  // allocate memory for grid
  const int memsize = width*height;
  x = new float [memsize];
  y = new float [memsize];

  // generate grid
  generategrid(x,y,Xmin,Xmax,Ymin,Ymax);

  // allocate solution memory + set it to zero
  solution = new float [memsize];
  memset(solution,0,memsize*sizeof(float));

  // set boundary conditions
  setBC(solution,x,y);

  cl_mem odd = clCreateBuffer(context_mm, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, memsize*sizeof(float), solution, &errVal);
  errorFunc(errVal, "odd clCreateBuffer");
  cl_mem even = clCreateBuffer(context_mm, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, memsize*sizeof(float), solution, &errVal);
  errorFunc(errVal, "even clCreateBuffer");

  cl_program program_mm;
	std::ifstream kernelFile("sorkernel.cl",std::ios::in);
	if(!kernelFile.is_open()){
		std::cout<< "Cannot Open the mentioned Kernel File for reading"<< std::endl;
		return 0;
	}

	std::ostringstream outputStream;
	outputStream << kernelFile.rdbuf();

	std::string sourceName = outputStream.str();
	const char *source = sourceName.c_str();

	program_mm = clCreateProgramWithSource(context_mm, 1, (const char**)&source, NULL, &errVal);

	if(program_mm == NULL){
		std::cout<< "Error in Program Source"<< std::endl;
	}

	errVal = clBuildProgram(program_mm, 0, NULL, NULL, NULL, NULL);

	if(errVal!=CL_SUCCESS){
		char buildLog[16384];
		clGetProgramBuildInfo(program_mm, deviceID[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program_mm);
		return NULL;
	}

	cl_kernel odd_mm = clCreateKernel(program_mm, "solve_odd", &errVal);
	errorFunc(errVal, "odd_mm clCreateKernel");

	cl_kernel even_mm = clCreateKernel(program_mm, "solve_even", &errVal);
	errorFunc(errVal, "even_mm clCreateKernel");

	cl_kernel oddeven_mm = clCreateKernel(program_mm, "merge_oddeven", &errVal);
	errorFunc(errVal, "oddeven_mm clCreateKernel");

	size_t local_ws[2] = {16, 16};
	size_t global_ws[2] = {800, 800};

	double start_time = clock();

	for(int i=0; i<itmax; i++){
		
		errVal = clSetKernelArg(odd_mm, 0, sizeof(cl_mem), &odd);
		errorFunc(errVal, "odd_mm clSetKernelArg 0");
		errVal = clSetKernelArg(odd_mm, 1, sizeof(cl_mem), &even);
		errorFunc(errVal, "odd_mm clSetKernelArg 1");
		errVal = clSetKernelArg(odd_mm, 2, sizeof(cl_int), &height);
		errorFunc(errVal, "odd_mm clSetKernelArg 2");
		errVal = clSetKernelArg(odd_mm, 3, sizeof(cl_int), &width);
		errorFunc(errVal, "odd_mm clSetKernelArg 3");
		errVal = clSetKernelArg(odd_mm, 4, sizeof(cl_int), &beta);
		errorFunc(errVal, "odd_mm clSetKernelArg 4");
		errVal = clSetKernelArg(odd_mm, 5, sizeof(cl_int), &omega);
		errorFunc(errVal, "odd_mm clSetKernelArg 5");
		
		errVal = clEnqueueNDRangeKernel(queue, odd_mm, 2, 0, global_ws, local_ws, 0, NULL, NULL);
		errorFunc(errVal, "odd_mm clEnqueueNDRangeKernel");
		errVal = clFinish(queue);
		errorFunc(errVal, "odd_mm clFinish");
		
		errVal = clSetKernelArg(even_mm, 0, sizeof(cl_mem), &odd);
		errorFunc(errVal, "even_mm clSetKernelArg 0");
		errVal = clSetKernelArg(even_mm, 1, sizeof(cl_mem), &even);
		errorFunc(errVal, "odd_mm clSetKernelArg 1");
		errVal = clSetKernelArg(even_mm, 2, sizeof(cl_int), &height);
		errorFunc(errVal, "odd_mm clSetKernelArg 2");
		errVal = clSetKernelArg(even_mm, 3, sizeof(cl_int), &width);
		errorFunc(errVal, "odd_mm clSetKernelArg 3");
		errVal = clSetKernelArg(even_mm, 4, sizeof(cl_int), &beta);
		errorFunc(errVal, "odd_mm clSetKernelArg 4");
		errVal = clSetKernelArg(even_mm, 5, sizeof(cl_int), &omega);
		errorFunc(errVal, "odd_mm clSetKernelArg 5");


		errVal = clEnqueueNDRangeKernel(queue, even_mm, 2, 0, global_ws, local_ws, 0, NULL, NULL);
		errorFunc(errVal, "even_mm clEnqueueNDRangeKernel");
		errVal = clFinish(queue);
		errorFunc(errVal, "even_mm clFinish");
		
	}

		errVal = clSetKernelArg(oddeven_mm, 0, sizeof(cl_mem), &odd);
		errorFunc(errVal, "oddeven_mm clSetKernelArg 0");
		errVal = clSetKernelArg(oddeven_mm, 1, sizeof(cl_mem), &even);
		errorFunc(errVal, "oddeven_mm clSetKernelArg 1");
		errVal = clSetKernelArg(oddeven_mm, 2, sizeof(cl_int), &height);
		errorFunc(errVal, "oddeven_mm clSetKernelArg 2");
		errVal = clSetKernelArg(oddeven_mm, 3, sizeof(cl_int), &width);
		errorFunc(errVal, "oddeven_mm clSetKernelArg 3");

		errVal = clEnqueueNDRangeKernel(queue, oddeven_mm, 2, 0, global_ws, local_ws, 0, NULL, NULL);
		errorFunc(errVal, "oddeven_mm clEnqueueNDRangeKernel");
		errVal = clFinish(queue);
		errorFunc(errVal, "oddeven_mm clFinish");

		double end_time = clock();
		double processing_time = (end_time-start_time)/CLOCKS_PER_SEC;
		std::cout<< "OpenCL computation time: "<< processing_time<<" seconds"<<std::endl;

		float *C;
		C = new float[memsize];
		memset(C,0,memsize*sizeof(float));
		errVal = clEnqueueReadBuffer(queue, odd, CL_TRUE, 0, memsize*sizeof(float), C, 0, NULL, NULL);
		errorFunc(errVal, "EnqueueReadBuffer");

		write_output(C);
		// Freeing the memory
		clReleaseMemObject(odd);
		clReleaseMemObject(even);
		clReleaseCommandQueue(queue);
		clReleaseKernel(odd_mm);
		clReleaseKernel(even_mm);
		clReleaseKernel(oddeven_mm);
		clReleaseProgram(program_mm);
		clReleaseContext(context_mm);

		delete [] x;
		delete [] y;
		delete [] solution;
		delete [] C;

		std::cout<< "Completed" << std::endl;

return 0;
}



void generategrid(float* x,float* y,const float Xmin,const float Xmax,const float Ymin,const float Ymax){

  float dx = fabs(Xmax-Xmin)/(width-1);
  float dy = fabs(Ymax-Ymin)/(height-1);

  for(size_t i = 0; i < width; i++){
      for(size_t j = 0; j < height; j++){
          x[i*height + j] = Xmin + i*dx;
          y[i*height + j] = Ymin + j*dy;
      } 
  }
}


void setBC(float* solution,const float* x, const float* y){

  for(size_t i = 0; i < width; i++){
      for(size_t j = 0; j < height; j++){

          size_t index = i*height + j ;
          if(i == 0){
              solution[index] = 0;
          }
          if(i == width-1){
              solution[index] = 0;
          }
          if(j == 0){
              solution[index] = sin(M_PI*x[index]); //bottom boundary condition
          }
          if(j == height-1){
              solution[index] = sin(M_PI*x[index])*exp(-M_PI); //top boundary condition
          }
      }
  }
}

void write_output(float* solution){

  std::ofstream file("clsolution.dat");
    for(int i = 0; i < width; i++){
      for(int j = 0; j < height; j++){
        file << std::setw(12) << solution[i*height + j];
		
      }
      file << std::endl;
	  
    }
    file.close();
}

void errorFunc(cl_int errVal, char *ss){
	if(errVal != CL_SUCCESS){
		std::cout<< "Error in "<< ss<<" "<< errVal<< std::endl;
		exit(EXIT_FAILURE);
	}
}
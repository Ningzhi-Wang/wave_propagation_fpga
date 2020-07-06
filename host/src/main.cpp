#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

typedef struct wave_model_2d
{
    cl_int nx;
    cl_int nz;
    cl_int dx;
    cl_float dt;
} WAVE_MODLE_2D;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 1;
cl_device_id device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue; // num_devices elements
cl_program program = NULL;
cl_kernel kernel; // num_devices elements

// Problem data.
scoped_aligned_ptr<float>  velocity, density, src, abs_facts;
scoped_array<scoped_aligned_ptr<float> > fields(3);
scoped_aligned_ptr<float> output; 


cl_mem velocity_buff, density_buff, abs_facts_buff;
cl_mem prev_buff, curr_buff, next_buff; 
cl_mem output_buff, source_buff;

bool use_fast_emulator = false;

//problem constants
int dx = 7;
int abs_layer_coefficient = 5;
int pad_size = ceil(dx * abs_layer_coefficient);
int nx = 501 + 2*pad_size;
int nz = 351 + pad_size + 3;
int receiver_depth = 73;
int sx = 150 + pad_size;
int sz = 83;
 
float frequency = 10.0;
float total_time = 2.0;
float source_amplitude = 1.0;
float courant_number = 0.3;
float abs_fact = 0.4;
float dt = courant_number * dx / 3500.0;
int time_steps = floor(total_time / dt);

WAVE_MODLE_2D model;

// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);


  // Optional argument to specify whether the fast emulator should be used.
  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  // Requires the number of devices to be known.
  init_problem();

  // Run the kernel.
  run();

  // Free the resources allocated
  cleanup();

  return 0;
}

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
  checkError(status, "Failed to get device");
  if (num_devices != 1) {
      printf("ERROR: There should be only one device in use.\n");
      return false;
  }
  printf("Platform: %s\n", getPlatformName(platform).c_str());

  // Create the context.
  context = clCreateContext(NULL, num_devices, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("wave_propagation", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");


  // Create per-device objects.
    // Command queue.
  queue = clCreateCommandQueue(context, device, 0, &status);
  checkError(status, "Failed to create command queue");

  // Kernel.
  const char *kernel_name = "wave_propagation";
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");


  velocity_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*nx*(nz-3), NULL, &status);
  checkError(status, "Failed to create velocity buffer");

  density_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*nx*nz, NULL, &status);
  checkError(status, "Failed to create density buffer");

  abs_facts_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*nx*(nz-3), NULL, &status);
  checkError(status, "Failed to create absorb factors buffer");

  prev_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*nx*nz, NULL, &status);
  checkError(status, "Failed to create wavefield buffer for previous time stamp");
  
  curr_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*nx*nz, NULL, &status);
  checkError(status, "Failed to create wavefield buffer for current time stamp");

  next_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nx*nz, NULL, &status);
  checkError(status, "Failed to create wavefield buffer for next time stamp");


  output_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nx*time_steps, NULL, &status);
  checkError(status, "Failed to create output buffer");

  source_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*time_steps, NULL, &status);
  checkError(status, "Failed to create output buffer");
  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }
  output.reset(time_steps * nx);

  velocity.reset(nx * nz);
  density.reset(nx * nz);
  abs_facts.reset(nx * nz);
  fields[0].reset(nx * nz);
  fields[1].reset(nx * nz);
  fields[2].reset(nx * nz);
  src.reset(time_steps);

  for (int i = 0; i < nz; ++ i) {
    for (int j = 0; j < nx; ++ j) {
      float vel;
      if (i >= 133 && i < 143 && j >= 220+pad_size && j < 230+pad_size) {
        vel = 3500.0;
      }else if (i > 113) {
        int fact = std::min(i-113, 160);
        vel = 1750.0 + fact * 10.0;
      }else {
        vel = 1500.0;
      }

      velocity[i*nx+j] = vel;

      if (vel > 1650.0) {
        density[i*nx+j] = 1.0 / (0.23 * pow(vel, 0.25));
      } else if (vel < 1500) {
        density[i*nx+j] = 1.0;
      } else {
        float ratio = (vel-1500.0)/150.0;
        density[i*nx+j] = 1.0 / (ratio * 0.23 * pow(vel, 0.25) + (1-ratio) * 1.0);
      }

      float abs = std::max(std::max(pad_size-j, j-pad_size-500), std::max(i-353, 0));
      abs = pow(abs, 2) / pow(pad_size, 2);
      abs = abs * vel * dt / dx * abs_fact;
      abs_facts[i*nx+j] = abs;

      fields[0][i*nx+j] = 0;
      fields[1][i*nx+j] = 0;
      fields[2][i*nx+j] = 0;
    }
  }

  int ns = 2.1 / (frequency * dt) + 0.9999;
  float ts = ns * dt;
  float a2 = pow(frequency * M_PI, 2);
  float t0 = ts/2 - dt/2;
  float max_source = 0;
  for (int i = 0; i < time_steps; ++i) {
      if (i < ns) {
        float at2 = a2*pow(i*dt-t0, 2);
        src[i] = source_amplitude * (1-2*at2) * exp(-at2);
        if (src[i] > max_source) {
          max_source = src[i];
        }
      }else {
        src[i] = 0.0;
      }
  }

  model.nx = (cl_int) nx;
  model.nz = (cl_int) nz;
  model.dx = (cl_int) dx;
  model.dt = (cl_float) dt;
}

void run() {
  cl_int status;

  int prev_idx = 0;
  int curr_idx = 1;
  int next_idx = 2;

  FILE *log;
  log = fopen("./output.log", "w");

  const double start_time = getCurrentTimestamp();

  int buffer_num = 7;
  cl_event write_event[buffer_num];
  float zero_pattern = 0;

  status = clEnqueueWriteBuffer(queue, velocity_buff, CL_FALSE,
      0, nx*(nz-3)*sizeof(float), velocity, 0, NULL, &write_event[0]);
  checkError(status, "Failed to transfer velocity");

  status = clEnqueueWriteBuffer(queue, abs_facts_buff, CL_FALSE,
      0, nx*(nz-3)*sizeof(float), abs_facts, 0, NULL, &write_event[1]);
  checkError(status, "Failed to transfer absorb factors");

  status = clEnqueueWriteBuffer(queue, density_buff, CL_FALSE,
      0, nx*nz*sizeof(float), density, 0, NULL, &write_event[2]);
  checkError(status, "Failed to transfer density");

  status = clEnqueueFillBuffer(queue, prev_buff, &zero_pattern,
      sizeof(float), 0, nz*nx*sizeof(float), 0, NULL, &write_event[3]);
  checkError(status, "Failed to initial previous field");

  status = clEnqueueFillBuffer(queue, curr_buff, &zero_pattern,
      sizeof(float), 0, nz*nx*sizeof(float), 0, NULL, &write_event[4]);
  checkError(status, "Failed to initial current field");

  status = clEnqueueFillBuffer(queue, next_buff, &zero_pattern,
      sizeof(float), 0, nz*nx*sizeof(float), 0, NULL, &write_event[5]);
  checkError(status, "Failed to initial next field");

  status = clEnqueueWriteBuffer(queue, source_buff, CL_FALSE,
      0, time_steps*sizeof(float), src, 0, NULL, &write_event[6]);
  checkError(status, "Failed to transfer source");

  cl_event kernel_event;
  cl_event finish_event;


  unsigned argi = 0;

  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), &nz);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), &dx);
  checkError(status, "Failed to set argument %d", argi - 1);

  int source_loc = sx+sz*nx;
  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), &source_loc);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), &receiver_depth);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), &time_steps);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_float), &dt);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &velocity_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &density_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &abs_facts_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &source_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &prev_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &curr_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &next_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buff);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clEnqueueTask(queue, kernel, buffer_num, write_event, &kernel_event);
  checkError(status, "Failed to launch kernel");

  status = clEnqueueReadBuffer(queue, output_buff, CL_FALSE,
        0,  nx*time_steps*sizeof(cl_float), output, 1, &kernel_event, &finish_event);
  
  // Release local events.
  for (int event_number = 0; event_number < buffer_num; ++event_number) {
    clReleaseEvent(write_event[event_number]);
  }

  // Wait for all devices to finish.
 
  clWaitForEvents(num_devices, &finish_event);
  clReleaseEvent(kernel_event);
  clReleaseEvent(finish_event);


  const double end_time = getCurrentTimestamp();
  fprintf(log, "Total time usage: %0.3f\n", (end_time-start_time));
  fclose(log);

  FILE* fout;
  fout = fopen("./fpga.csv", "w");
  fwrite(output, sizeof(float), nx*time_steps, fout);
  fclose(fout);
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) {
    clReleaseKernel(kernel);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }

  if(velocity_buff) 
    clReleaseMemObject(velocity_buff);
  
  if(density_buff) 
    clReleaseMemObject(density_buff);

  if(abs_facts_buff) 
    clReleaseMemObject(abs_facts_buff);

  if(prev_buff) 
    clReleaseMemObject(prev_buff);

  if(curr_buff) 
    clReleaseMemObject(curr_buff);
    
  if(next_buff) 
    clReleaseMemObject(next_buff);

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}
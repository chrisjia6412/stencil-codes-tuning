// Copyright (C) 2013-2014 Altera Corporation, San Jose, California, USA. All rights reserved. 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this 
// software and associated documentation files (the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, merge, 
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to 
// whom the Software is furnished to do so, subject to the following conditions: 
// The above copyright notice and this permission notice shall be included in all copies or 
// substantial portions of the Software. 
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
// OTHER DEALINGS IN THE SOFTWARE. 
//  
// This agreement shall be governed in all respects by the laws of the State of California and 
// by the laws of the United States of America. 


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include "convolution.h"

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
scoped_array<cl_mem> input_a_buf; // num_devices elements
scoped_array<cl_mem> input_b_buf; // num_devices elements
scoped_array<cl_mem> output_buf; // num_devices elements

// Problem data.
const unsigned A_height = 1; //padding the src matrix by adding the height and width with filter matrix height and width
const unsigned A_width  = 1024 * 1024;
const unsigned B_height = 1;
const unsigned B_width  = 17;
const unsigned C_height = 1;
const unsigned C_width  = 1024 * 1024;

scoped_array<scoped_aligned_ptr<float> > input_a; // num_devices elements
scoped_aligned_ptr<float> input_b;
scoped_array<scoped_aligned_ptr<float> > output; // num_devices elements
scoped_array<float> ref_output;
scoped_array<unsigned> rows_per_device; // num_devices elements

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void compute_reference();
void verify();
void cleanup();

// Entry point.
int main() {
  printf("Matrix sizes:\n  A: %d x %d\n  B: %d x %d\n  C: %d x %d\n",
      A_height, A_width, B_height, B_width, C_height, C_width);


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

/////// HELPER FUNCTIONS ///////

// Randomly generate a float number between -10 and 10.
float rand_float() {
    return (rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("convolution", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  rows_per_device.reset(num_devices);
  input_a_buf.reset(num_devices);
  input_b_buf.reset(num_devices);
  output_buf.reset(num_devices);

  const unsigned num_block_rows = C_width / BLOCK_SIZE;

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "convolution";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of rows processed by this device.
    // First do this computation in block-rows.
    rows_per_device[i] = num_block_rows / num_devices; // this is the number of block-rows

    // Spread out the remainder of the block-rows over the first
    // N % num_devices.
    if(i < (num_block_rows % num_devices)) {
      rows_per_device[i]++;
    }

    // Multiply by BLOCK_SIZE to get the actual number of rows.
    rows_per_device[i] *= BLOCK_SIZE;

    input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        A_height * A_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        B_height * B_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        C_height * C_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
  }

  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  printf("Generating input matrices\n");
  input_a.reset(num_devices);
  output.reset(num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    input_a[i].reset(A_height * A_width);
    output[i].reset(C_height * C_width);

    // printf("array A elements\n");
    for(unsigned j = 0; j < A_height * A_width; ++j) {
      input_a[i][j] = rand_float();
    }
  }

  input_b.reset(B_height * B_width);
  // printf("array B elements\n");
  for(unsigned i = 0; i < B_height * B_width; ++i) {
    input_b[i] = rand_float();
  }
}

void run() {
  cl_int status;

  // Transfer inputs to each device. Each of the host buffers supplied to
  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  for(unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, A_height * A_width * sizeof(float), input_a[i], 0, NULL, NULL);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, B_width * B_height * sizeof(float), input_b, 0, NULL, NULL);
    checkError(status, "Failed to transfer input B");
  }

  // Wait for all queues to finish.
  for(unsigned i = 0; i < num_devices; ++i) {
    clFinish(queue[i]);
  }

  // Launch kernels.
  // This is the portion of time that we'll be measuring for throughput
  // benchmarking.
  scoped_array<cl_event> kernel_event(num_devices);

  const double start_time = getCurrentTimestamp();
  for(unsigned i = 0; i < num_devices; ++i) {
    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(C_width), &C_width);
    checkError(status, "Failed to set argument %d", argi - 1);



    status = clEnqueueTask(queue[i], kernel[i], 0, NULL, &kernel_event[i]);
    checkError(status, "Failed to launch kernel");
  }

  // Wait for all kernels to finish.
  clWaitForEvents(num_devices, kernel_event);

  const double end_time = getCurrentTimestamp();
  const double total_time = end_time - start_time;

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", total_time * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }


  // Release kernel events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
  }

  // Read the result.
  for(unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_TRUE,
        0, C_height * C_width * sizeof(float), output[i], 0, NULL, NULL);
    checkError(status, "Failed to read output matrix");
  }

  // Verify results.
  compute_reference();
  verify();
}

void compute_reference() {
  // Compute the reference output.
  printf("Computing reference output\n");
  ref_output.reset(C_height * C_width);

  for(unsigned x = 0; x < C_width; ++x) {
    float sum = 0.0f; 
    for(unsigned m = 0; m < B_width; ++m) {
      if(x-m >= 0)
        sum += input_a[0][x-m] * input_b[m];
    }
      
    ref_output[x] = sum;
  }
  
}

void verify() {
  printf("Verifying\n");

  float diff = 0.0f;
  float ref = 0.0f;
  for(unsigned x = 0; x < C_width; ++x) {
    const int o = output[0][x];
    const int r = ref_output[x];
    const float d = o - r;
    diff += d*d;
    ref += r*r;   
  }
  
  const float diff_l2norm = sqrtf(diff);
  const float ref_l2norm = sqrtf(ref);
  const float error = diff_l2norm / ref_l2norm;
  const bool pass = error < 1e-3;

  printf("Verification: %s\n", pass ? "PASS" : "FAIL");
  if(!pass) {
    printf("Error (L^2-Norm): %0.3g\n", error);
  }
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }
  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}


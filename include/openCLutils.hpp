//
//  openCLutils.hpp
//  openCLutils
//
//  Created by Adrian Daniel Bodirlau on 02/11/2022.
//

#ifndef openCLutils_
#define openCLutils_

/* The classes below are exported */
#pragma GCC visibility push(default)
#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <string>
#import "CL/opencl.h"

const char* CLDisplayError(cl_int error);
bool CLError(cl_int error, int line);
cl_int CLGetPlatforms(cl_platform_id*& platforms);
char* CLGetPlatformInfo(cl_platform_id platform, cl_platform_info info);
cl_int CLGetDevices(const cl_platform_id platform, cl_device_id*& devices, cl_device_type type);
char* CLGetDeviceInfo(cl_device_id device, cl_device_info info);

std::string readFile(char* sourceFileName);

cl_int CLBuildProgram(cl_program* program,cl_device_id* device);
void CLGeneralCleanUp(cl_program& program, cl_kernel& kernel, cl_command_queue& commands, cl_context& context, cl_platform_id* platforms, cl_device_id* devices);


#pragma GCC visibility pop
#endif

//
//  openCLutils.cpp
//  openCLutils
//
//  Created by Adrian Daniel Bodirlau on 02/11/2022.
//

#include <iostream>
#include "openCLutils.hpp"
#include "openCLutilsPriv.hpp"
#include "CL/opencl.h"
#include <fstream>
#include <string>
#include <vector>

bool CLError(cl_int error, int line) {
    if(error != CL_SUCCESS) {
        printf("Error at line %d: %s\n", line,CLDisplayError(error));
        return true;
    }
    return false;
}

const char* CLDisplayError(cl_int error) {
    switch(error) {
        // Run-time and JIT compile errors
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

char* CLGetPlatformInfo(cl_platform_id platform, cl_platform_info info) {
    size_t valueSize;
    char* value;
    clGetPlatformInfo(platform, info, 0, NULL, &valueSize);
    value = static_cast<char*>(malloc(valueSize));
    clGetPlatformInfo(platform, info, valueSize, value, NULL);
    return value;
}


cl_int CLGetPlatforms(cl_platform_id*& platforms) {
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    return clGetPlatformIDs(numPlatforms, platforms, NULL);
}

cl_int CLGetDevices(const cl_platform_id platform, cl_device_id*& devices, cl_device_type type) {
    cl_uint numDevices;
    clGetDeviceIDs(platform, type, 0, NULL, &numDevices);
    devices = new cl_device_id[numDevices];
    return clGetDeviceIDs(platform, type, numDevices, devices, NULL);
}

char* CLGetDeviceInfo(cl_device_id device, cl_device_info info) {
    size_t valueSize;
    char* value;
    clGetDeviceInfo(device, info, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, info, valueSize, value, NULL);
    return  value;
}
std::string readFile(char* sourceFileName) {
    std::ifstream streamSourceFile(sourceFileName);
    std::string stringSourceFile((std::istreambuf_iterator<char>(streamSourceFile)), (std::istreambuf_iterator<char>()));
    streamSourceFile.close();
    return stringSourceFile;
}

cl_int CLBuildProgram(cl_program* program, cl_device_id* device){
    cl_int error;
    error = clBuildProgram(*program, 1, device, NULL, NULL, NULL);
    std::vector<char> buildLog;
    size_t logSize;
    clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    buildLog.resize(logSize);
    clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
    if(logSize < 1)
        std::cout<<&buildLog[0]<<std::endl;
    return error;
}

void CLGeneralCleanUp(cl_program& program, cl_kernel& kernel, cl_command_queue& commands, cl_context& context, cl_platform_id* platforms, cl_device_id* devices){
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(platforms);
    delete [] devices;
}

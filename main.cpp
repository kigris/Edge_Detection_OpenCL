//
//  main.cpp
//  Lab3
//
//  Created by Adrian Daniel Bodirlau on 02/11/2022.
// Sobel and canny edge detection

#include <iostream>
#include "CL/bmpfuncs.c"
#include "CL/openCLutils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <time.h>
using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    cl_platform_id* platforms{nullptr};
    CLGetPlatforms(platforms);
    cl_device_id* devices;
    CLGetDevices(platforms[0], devices, CL_DEVICE_TYPE_GPU);
    cl_device_id device = devices[0];
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue commands = clCreateCommandQueue(context, device, NULL, NULL);

    // Program
    string programSourceStr{readFile((char*)"Lab3/source.CL")};
    const char* programSource = programSourceStr.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, NULL, NULL);
    CLBuildProgram(&program, &device);

//     Old method for reading image
//    int width{0};
//    int height{0};

//    float* img = readImage((char*)"Lab3/input.bmp", &width, &height);
    Mat img = imread("Lab3/img2.jpg", 0);
    img.convertTo(img, CV_32F);
//    // Image read
//    Mat mat = imread("Lab3/input.bmp",0);
//    cv::Mat flat = mat.reshape(1, 1);
////    std::vector<uchar> vec = mat.isContinuous()? flat : flat.clone();
//
//    Mat imgF;
//    flat.convertTo(imgF, CV_32F);
//    std::vector<uchar> vec = mat.isContinuous()? imgF : imgF.clone();
////    vector<float> img = imgF.data;
//    vector<float> img;
//    for(auto e: vec) {
//        img.push_back(e);
//    }
//
    int width{img.cols};
    int height{img.rows};

//    Mat img2(mat.rows, mat.cols, mat.type(), vec.data());
//
//    namedWindow("peter", WINDOW_AUTOSIZE);
//    imshow("peter", img2);
//    waitKey(0);
    
    cl_image_format imgFormat;
    imgFormat.image_channel_order = CL_R;
    imgFormat.image_channel_data_type = CL_FLOAT;
    cl_image_desc imgDesc;
    imgDesc.image_width=width;
    imgDesc.image_height=height;
    imgDesc.image_type=CL_MEM_OBJECT_IMAGE2D;
    cl_mem imgInBuff = clCreateImage(context, CL_MEM_READ_ONLY, &imgFormat, &imgDesc, NULL, NULL);
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, NULL);
    size_t origin[3]{0,0,0};
    size_t region[3]{(size_t)width,(size_t)height,1};
    // Setting the size of the workers
    size_t globalWorkers[2]{(size_t)width,(size_t)height};
    
    // Writing the input image
    clEnqueueWriteImage(commands, imgInBuff, CL_TRUE, origin, region, 0, 0, img.ptr<float>(), 0, NULL, NULL);
    
    // Output for gaussian blur
    cl_mem imgGaussBuff = clCreateImage(context, CL_MEM_READ_WRITE, &imgFormat, &imgDesc, NULL, NULL);
    
    // Gaussian blur mask
    float gaussMask[9]{0.0625f,0.125f,0.0625f,0.1250f,0.250f,0.1250f,0.0625f,0.125f,0.0625f};
    float maskSum = 1.0f;
    float gaussMaskMult = 40.0f;
    int maskGaussWidth=sizeof(gaussMask)/sizeof(gaussMask[0]);
    int isMaskGaussAvg=1;
    cl_mem maskGaussBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(gaussMask), NULL, NULL);
    
    // Writing the mask for gaussian blur
    clEnqueueWriteBuffer(commands, maskGaussBuff, CL_TRUE, 0, sizeof(gaussMask), gaussMask, 0, NULL, NULL);
    
    // Gaussian blurring the magnitude pixel
    cl_kernel kernelGauss = clCreateKernel(program, "maskPixel", NULL);
    clSetKernelArg(kernelGauss, 0, sizeof(cl_mem), &imgInBuff);
    clSetKernelArg(kernelGauss, 1, sizeof(cl_mem), &imgGaussBuff);
    clSetKernelArg(kernelGauss, 2, sizeof(cl_sampler), &sampler);
    clSetKernelArg(kernelGauss, 3, sizeof(int), &maskGaussWidth);
    clSetKernelArg(kernelGauss, 4, sizeof(cl_mem), &maskGaussBuff);
    clSetKernelArg(kernelGauss, 5, sizeof(float), &maskSum);
    clSetKernelArg(kernelGauss, 6, sizeof(float), &gaussMaskMult);
    clSetKernelArg(kernelGauss, 7, sizeof(int), &isMaskGaussAvg);
    clFinish(commands);
    clock_t begin = clock();
    // Executing gaussian blur
    clEnqueueNDRangeKernel(commands, kernelGauss, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    clock_t end = clock();
    double execTime = (double)(end-begin)/CLOCKS_PER_SEC;
    cout<<"Excution time: "<<execTime<<endl;
    
    // Mask for vertical edge detection
    float maskY[9]{-1,0,1,-2,0,2,-1,0,1};
    float maskYMult=1.0f;
    int maskYWidth = sizeof(maskY)/sizeof(maskY[0]);
    int isMaskYAvg = 0;
    cl_mem maskYBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(maskY), NULL, NULL);

    // Writing the image and the mask for vertical edge detection
    clEnqueueWriteBuffer(commands, maskYBuff, CL_TRUE, 0, sizeof(maskY), maskY, 0, NULL, NULL);
    
    // Output for vertical detection
    cl_mem imgYOutBuff = clCreateImage(context, CL_MEM_READ_WRITE, &imgFormat, &imgDesc, NULL, NULL);
    
    // Detect vertical edges
    cl_kernel kernelY = clCreateKernel(program, "maskPixel", NULL);
    clSetKernelArg(kernelY, 0, sizeof(cl_mem), &imgGaussBuff);
    clSetKernelArg(kernelY, 1, sizeof(cl_mem), &imgYOutBuff);
    clSetKernelArg(kernelY, 2, sizeof(cl_sampler), &sampler);
    clSetKernelArg(kernelY, 3, sizeof(int), &maskYWidth);
    clSetKernelArg(kernelY, 4, sizeof(cl_mem), &maskYBuff);
    clSetKernelArg(kernelY, 5, sizeof(float), &maskSum);
    clSetKernelArg(kernelY, 6, sizeof(float), &maskYMult);
    clSetKernelArg(kernelY, 7, sizeof(int), &isMaskYAvg);
    
    // Executing the horizontal edge detection kernel
    clEnqueueNDRangeKernel(commands, kernelY, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    
    // Mask for horizontal edge detection
    float maskX[9]{-1,-2,-1,0,0,0,1,2,1};
    float maskXMult=1.0f;
    int maskXWidth = sizeof(maskX)/sizeof(maskX[0]);
    int isMaskXAvg = 0;
    cl_mem maskXBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(maskX), NULL, NULL);
    
    // Writing the mask for horizontal edge detection
    clEnqueueWriteBuffer(commands, maskXBuff, CL_TRUE, 0, sizeof(maskX), maskX, 0, NULL, NULL);
    
    // Output for horizontal detection
    cl_mem imgXOutBuff = clCreateImage(context, CL_MEM_READ_WRITE, &imgFormat, &imgDesc, NULL, NULL);
    
    // Detecting horizontal edges
    cl_kernel kernelX = clCreateKernel(program, "maskPixel", NULL);
    clSetKernelArg(kernelX, 0, sizeof(cl_mem), &imgGaussBuff);
    clSetKernelArg(kernelX, 1, sizeof(cl_mem), &imgXOutBuff);
    clSetKernelArg(kernelX, 2, sizeof(cl_sampler), &sampler);
    clSetKernelArg(kernelX, 3, sizeof(int), &maskXWidth);
    clSetKernelArg(kernelX, 4, sizeof(cl_mem), &maskXBuff);
    clSetKernelArg(kernelX, 5, sizeof(float), &maskSum);
    clSetKernelArg(kernelX, 6, sizeof(float), &maskXMult);
    clSetKernelArg(kernelX, 7, sizeof(int), &isMaskXAvg);
    
    // Executing the horizontal edge detection kernel
    clEnqueueNDRangeKernel(commands, kernelX, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    
    // Buffer for storing the magnitude image
    cl_mem imgMagnBuff = clCreateImage(context, CL_MEM_READ_WRITE, &imgFormat, &imgDesc, NULL, NULL);
    
    // Getting the magnitude
    cl_kernel kernelMagn = clCreateKernel(program, "avgPixel", NULL);
    clSetKernelArg(kernelMagn, 0, sizeof(cl_mem), &imgYOutBuff);
    clSetKernelArg(kernelMagn, 1, sizeof(cl_mem), &imgXOutBuff);
    clSetKernelArg(kernelMagn, 2, sizeof(cl_mem), &imgMagnBuff);
    clSetKernelArg(kernelMagn, 3, sizeof(cl_sampler), &sampler);
    
    // Executing the magnitutde edge detection kernel
    clEnqueueNDRangeKernel(commands, kernelMagn, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    
    // Buffer for storing the angle image
    cl_mem imgAngBuff = clCreateImage(context, CL_MEM_READ_WRITE, &imgFormat, &imgDesc, NULL, NULL);
    
    // Getting the angle
    cl_kernel kernelAng = clCreateKernel(program, "anglePixel", NULL);
    clSetKernelArg(kernelAng, 0, sizeof(cl_mem), &imgYOutBuff);
    clSetKernelArg(kernelAng, 1, sizeof(cl_mem), &imgXOutBuff);
    clSetKernelArg(kernelAng, 2, sizeof(cl_mem), &imgAngBuff);
    clSetKernelArg(kernelAng, 3, sizeof(cl_sampler), &sampler);
    
    // Executing the angle detection kernel
    clEnqueueNDRangeKernel(commands, kernelAng, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    
    // Output for canny kernel
    cl_mem localMaxBuff = clCreateImage(context, CL_MEM_READ_WRITE, &imgFormat, &imgDesc, NULL, NULL);
    
    // Max explore area detection
    int maxExpArea = 3;
    int angleExp = 0;
    
    // Canny kernel
    cl_kernel cannyKern = clCreateKernel(program, "cannyLocalMax", NULL);
    clSetKernelArg(cannyKern, 0, sizeof(cl_mem), &imgMagnBuff);
    clSetKernelArg(cannyKern, 1, sizeof(cl_mem), &imgAngBuff);
    clSetKernelArg(cannyKern, 2, sizeof(cl_mem), &localMaxBuff);
    clSetKernelArg(cannyKern, 3, sizeof(int), &maxExpArea);
    clSetKernelArg(cannyKern, 4, sizeof(int), &angleExp);
    clSetKernelArg(cannyKern, 5, sizeof(cl_sampler), &sampler);
    
    // Executing canny kernel
    clEnqueueNDRangeKernel(commands, cannyKern, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    
    // Output for canny kernel hysteresis
    cl_mem hystBuff = clCreateImage(context, CL_MEM_READ_WRITE, &imgFormat, &imgDesc, NULL, NULL);
    
    float lowThr=15.0f;
    float highThr=50.0f;
    
    // Canny kernel
    cl_kernel hystKern = clCreateKernel(program, "cannyHyst", NULL);
    clSetKernelArg(hystKern, 0, sizeof(cl_mem), &localMaxBuff);
    clSetKernelArg(hystKern, 1, sizeof(cl_mem), &hystBuff);
    clSetKernelArg(hystKern, 2, sizeof(cl_sampler), &sampler);
    clSetKernelArg(hystKern, 3, sizeof(float), &lowThr);
    clSetKernelArg(hystKern, 4, sizeof(float), &highThr);
    
    // Executing canny kernel
    clEnqueueNDRangeKernel(commands, hystKern, 2, NULL, globalWorkers, NULL, 0, NULL, NULL);
    clFinish(commands);
    
    // Reading all the images out
    float* imgYOut = (float*)malloc(sizeof(float)*width*height);
    clEnqueueReadImage(commands, imgYOutBuff, CL_TRUE, origin, region, 0, 0, imgYOut, 0, NULL, NULL);
    float* imgXOut = (float*)malloc(sizeof(float)*width*height);
    clEnqueueReadImage(commands, imgXOutBuff, CL_TRUE, origin, region, 0, 0, imgXOut, 0, NULL, NULL);
    float* imgMagnOut = (float*)malloc(sizeof(float)*width*height);
    clEnqueueReadImage(commands, imgMagnBuff, CL_TRUE, origin, region, 0, 0, imgMagnOut, 0, NULL, NULL);
    float* imgGaussOut = (float*)malloc(sizeof(float)*width*height);
    clEnqueueReadImage(commands, imgGaussBuff, CL_TRUE, origin, region, 0, 0, imgGaussOut, 0, NULL, NULL);
    float* imgAngOut = (float*)malloc(sizeof(float)*width*height);
    clEnqueueReadImage(commands, imgAngBuff, CL_TRUE, origin, region, 0, 0, imgAngOut, 0, NULL, NULL);
    float* imgLocalMaxOut = (float*)malloc(sizeof(float)*width*height);
    clEnqueueReadImage(commands, localMaxBuff, CL_TRUE, origin, region, 0, 0, imgLocalMaxOut, 0, NULL, NULL);
    float* imgHystOut = (float*)malloc(sizeof(float)*width*height);
    clEnqueueReadImage(commands, hystBuff, CL_TRUE, origin, region, 0, 0, imgHystOut, 0, NULL, NULL);
    
    // Storing the images
//    storeImage(imgYOut, "Lab3/VEdgeDetect.bmp", height, width, "Lab3/input.bmp");
//    storeImage(imgXOut, "Lab3/HEdgeDetect.bmp", height, width, "Lab3/input.bmp");
//    storeImage(imgMagnOut, "Lab3/edgeDetect.bmp", height, width, "Lab3/input.bmp");
//    storeImage(imgGaussOut, "Lab3/gaussBlur.bmp", height, width, "Lab3/input.bmp");
//    storeImage(imgAngOut, "Lab3/angleDetect.bmp", height, width, "Lab3/input.bmp");
//    storeImage(imgLocalMaxOut, "Lab3/localMax.bmp", height, width, "Lab3/input.bmp");
//    storeImage(imgHystOut, "Lab3/hyst.bmp", height, width, "Lab3/input.bmp");
    
    Mat imgOut{height,width,img.type(),imgHystOut};
    imwrite("Lab3/output.bmp", imgOut);
    
    // Cleaning
    CLGeneralCleanUp(program, kernelY, commands, context, platforms, devices);
    clReleaseKernel(kernelX);
    clReleaseKernel(kernelGauss);
    clReleaseKernel(hystKern);
    clReleaseKernel(kernelMagn);
    clReleaseKernel(kernelAng);
    return 0;
}

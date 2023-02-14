__kernel void maskPixel(__read_only image2d_t inputImage, __write_only image2d_t outputImage, sampler_t sampler, int maskSize, global float* mask, float maskSum, float maskMult, int isAverage){
    // Get thread id
    int col = get_global_id(0);
    int row = get_global_id(1);
    
    // Coords used later
    int2 coords={0,0};
    // Get square root of maskSize to get the width
    int maskWidth=(int)sqrt((float)maskSize);
    // Index loop mask width
    int hMaskW = (int)(maskWidth/2);
    // We are storing here the final result
    float4 sum={0.0f,0.0f,0.0f,0.0f};
    // Mask index
    int maskI=0;
    
    // Iterating over rows (height)
    for(int i=-hMaskW;i<=hMaskW;i++){
        coords.y=row+i; // Get the current row
        // Iterating over columns (width)
        for(int j=-hMaskW;j<=hMaskW;j++){
            coords.x=col+j; // Get the current column
            // Get sum of first channel x
            float4 pixel = read_imagef(inputImage,sampler,coords);
            sum.x+=pixel.x*mask[maskI]*maskMult;
            maskI++;
        }
    }
    
    // If we need to average
    if(isAverage==1){
        sum.x=sum.x/(maskSum*(maskMult/2));
        sum.x=min(255.0f,max(0.0f,sum.x));
    }
    
    // If we are not out of the bounds
    coords.x = col;
    coords.y = row;
    write_imagef(outputImage, coords, sum);
}

__kernel void avgPixel(__read_only image2d_t inputImage1, __read_only image2d_t inputImage2, __write_only image2d_t outputImage, sampler_t sampler) {
    // Get thread id
    int col = get_global_id(0);
    int row = get_global_id(1);
    
    // Coords for getting the pixel of each thread
    int2 coords={col,row};
    
    // Getting the pixels of both images
    float4 pixel1 = read_imagef(inputImage1,sampler,coords);
    float4 pixel2 = read_imagef(inputImage2,sampler,coords);
    
    // Getting the magnitude
    float4 output={0.0f,0.0f,0.0f,0.0f};
    
    output.x=min(255.0f,max(0.0f,hypot(pixel1.x,pixel2.x)));
    
    // Writing the output
    write_imagef(outputImage,coords,output);
}

__kernel void anglePixel(__read_only image2d_t inputImage1, __read_only image2d_t inputImage2, __write_only image2d_t outputImage, sampler_t sampler) {
    // Get thread id
    int col = get_global_id(0);
    int row = get_global_id(1);
    
    // Coords for getting the pixel of each thread
    int2 coords={col,row};
    
    // Getting the pixels of both images
    float4 pixel1 = read_imagef(inputImage1,sampler,coords);
    float4 pixel2 = read_imagef(inputImage2,sampler,coords);
    
    // Getting the magnitude
    float4 output={0.0f,0.0f,0.0f,0.0f};
    output.x = atan2(pixel1.x,pixel2.x);
    
    if (output.x < 0)
            output.x = fmod((output.x + 2*M_PI_F),(2*M_PI_F));
    
    output.x = ((int)(degrees(output.x * (M_PI_F/8) + M_PI_F/8) / 45) * 45) % 180;
    
    // Writing the output
    write_imagef(outputImage,coords,output);
    }

__kernel void cannyLocalMax(__read_only image2d_t sobelEdge, __read_only image2d_t sobelAngle, __write_only image2d_t outputImage, int exploreW, int angle, sampler_t sampler){
    // Get thread ids
    int col = get_global_id(0);
    int row = get_global_id(1);
    
//    int hWE = (int)exploreW/2;
    int2 coords={col,row};
    float4 currentP = read_imagef(sobelEdge,sampler,coords);
    float4 currentAngle = read_imagef(sobelAngle,sampler,coords);
    
    float4 compareP1={0.0f,0.0f,0.0f,0.0f};
    float4 compareP2={0.0f,0.0f,0.0f,0.0f};
    
    int2 compareCoords={col,row};
    
    switch((int)currentAngle.x){
        // If the current pixel has 0º angle it means it's going north-south
        // needs to be compared to east/west pixels
        case 0:
            // Move left to grab west pixel
            compareCoords.x-=1;
            compareP1=read_imagef(sobelEdge,sampler,compareCoords);
            // Move right ro grab east pixel
            compareCoords.x+=2;
            compareP2=read_imagef(sobelEdge,sampler,compareCoords);
            // If the compare pixels are bigger
            if((currentP.x<=compareP1.x)||(currentP.x<=compareP2.x))
                currentP.x=0;
            break;
        // If the current pixel has 45º angle it means it's going NW/SE
        // needs to be compared to NE/SW pixels
        case 45:
            // Move left and up to grab NE pixel
            compareCoords.x+=1;
            compareCoords.y-=1;
            compareP1=read_imagef(sobelEdge,sampler,compareCoords);
            // Move right and down ro grab SW pixel
            compareCoords.x-=2;
            compareCoords.y+=2;
            compareP2=read_imagef(sobelEdge,sampler,compareCoords);
            // If the compare pixels are bigger
            if((currentP.x<=compareP1.x)||(currentP.x<=compareP2.x))
                currentP.x=0;
            break;
        // If the current pixel has 90º angle it means it's going E/W
        // needs to be compared to north/south pixels
        case 90:
            // Move up to grab north pixel
            compareCoords.y-=1;
            compareP1=read_imagef(sobelEdge,sampler,compareCoords);
            // Move down ro grab south pixel
            compareCoords.y+=2;
            compareP2=read_imagef(sobelEdge,sampler,compareCoords);
            // If the compare pixels are bigger
            if((currentP.x<=compareP1.x)||(currentP.x<=compareP2.x))
                currentP.x=0;
            break;
        // If the current pixel has 135º angle it means it's going NE/SW
        // needs to be compared to NW/SE pixels
        case 135:
            // Move left and up to grab NW pixel
            compareCoords.x-=1;
            compareCoords.y-=1;
            compareP1=read_imagef(sobelEdge,sampler,compareCoords);
            // Move right and down ro grab SE pixel
            compareCoords.x+=2;
            compareCoords.y+=2;
            compareP2=read_imagef(sobelEdge,sampler,compareCoords);
            // If the compare pixels are bigger
            if((currentP.x<=compareP1.x)||(currentP.x<=compareP2.x))
                currentP.x=0;
            break;
        default:
            break;
    }
    

    write_imagef(outputImage,coords,currentP);
}

__kernel void cannyHyst(__read_only image2d_t inputImage, __write_only image2d_t outputImage, sampler_t sampler, float lowThr, float hightThr) {
    // Get thread ids
    int col = get_global_id(0);
    int row = get_global_id(1);
    int2 coords={col,row};
    
    const float EDGE = 255.0f;
    
    float4 magn = read_imagef(inputImage,sampler,coords);
    
    if(magn.x>=hightThr)
        magn.x=EDGE;
    else if(magn.x<=lowThr)
        magn.x=0;
    else{
        float med = (hightThr+lowThr)/2;
        
        if(magn.x>=med)
            magn.x=255;
        else
            magn.x=0;
    }
    write_imagef(outputImage,coords,magn);
}

// C libraries
#include <cstdlib>
#include <cstdio>

//Third party libraries for image manipulation
#define STB_IMAGE_IMPLEMENTATION
#include "stb_library/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_library/stb_image_write.h"

// CUDA libraries
#include <cuda_runtime.h>
#include <helper_cuda.h>

//Function to dinamically generate gaussian kernel
void generate_kernel(int size, double** kernel) 
{   
    //Standard deviation 
    double sigma = 15.0; 
    
    //Sum to normalize values later 
    double sum = 0.0; 
    
    int mid = size/2;

    //Generate kernel using exponentiall probability function
    for (int x = -mid; x <= mid; x++) { 
        for (int y = -mid; y <= mid; y++) { 
            kernel[x + mid][y + mid] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma); 
            sum += kernel[x + mid][y + mid]; 
        } 
    } 
  
    //Normalization of kernel for a cleaner blurring effect
    for (int i = 0; i < size; ++i) 
        for (int j = 0; j < size; ++j) 
            kernel[i][j] /= sum; 
} 

/*
Cuda kernel to apply blurring filter:
- img -> original image array
- blurred_img -> array of blurred image which is going to be modified
- global_kernel -> gaussian kernel
- kernel_size -> size of kernel
- n_pixels -> number of pixels that each thread is going to operate
- TOTAL_THREADS -> Total amount of threads that will be running
- width -> width in pixels of original image
- height -> height in pixels of original image
*/

__global__ void blurImage(unsigned char* img, unsigned char* blurred_img, double** global_kernel,
                         unsigned char kernel_size, int n_pixels, int TOTAL_THREADS,
                         int width, int height){
    
    
    //Thread 0 of each block uploads gaussian kernel from global memory into shared memory
    __shared__ double kernel[15][15];
    if(threadIdx.x == 0){
        for(unsigned char i = 0 ; i < kernel_size; ++i){
            for(unsigned char j = 0; j < kernel_size; ++j)
                kernel[i][j] = global_kernel[i][j]; 
        }
    }

    //Variables to store info needed for convolution
    unsigned char mid_size = (uint8_t)(kernel_size/2);
    int target_pixel;
    int current_pixel;
    int index = blockDim.x * blockIdx.x + threadIdx.x; 

    //Variables for channel info of neighbor pixels
    int pixel_valueBlue;
    int pixel_valueGreen;
    int pixel_valueRed;

    double valueRed;
    double valueGreen;
    double valueBlue;

    //All threads must wait for kernel to be fully loaded to access it
    __syncthreads();
        
    //Convolution of n_pixels per thread
    for(int p = 0; p < n_pixels; ++p) {

        current_pixel = index + p*TOTAL_THREADS;
        valueRed = valueGreen = valueBlue = 0;
        
        //Traverse each value of kernel matrix
        for(int i = -mid_size; i <= mid_size; ++i){
            for(int j = -mid_size; j <= mid_size; ++j){
                //Calculate neighbor pixel of image which is going to be taken
                target_pixel = current_pixel + (i*width) + j;

                //Extract each one of the RGB channeld of neighbor pixel
                pixel_valueRed = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*3)+0);
                pixel_valueGreen = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*3)+1);
                pixel_valueBlue = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*3)+2);

                //Sum of values from each channel multiplied with kernel elements
                valueRed += kernel[i+mid_size][j+mid_size] * pixel_valueRed;
                valueGreen += kernel[i+mid_size][j+mid_size] * pixel_valueGreen;
                valueBlue += kernel[i+mid_size][j+mid_size] * pixel_valueBlue;
            }
        }
        
         //Assign modified values to blurred image
         *(blurred_img + (current_pixel*3) + 0) =  (uint8_t)(valueRed);
         *(blurred_img + (current_pixel*3) + 1) =  (uint8_t)(valueGreen);
         *(blurred_img + (current_pixel*3) + 2) =  (uint8_t)(valueBlue);  
    }
}

int main(int argc, char* argv[]){

    //Print info of current device
    //Get Device cores and multiprocessors
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device specification (%s):\n", deviceProp.name);
    printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("CUDA Cores/MP: %d\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
    printf("Total amount of cores: %d\n",_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    
    
    int width, height, channels;

    //Ensure number of arguments is correct
    if(argc != 6) {
        perror("Wrong number of arguments!");
        return EXIT_FAILURE;
    }

    //Load image and it's data
    unsigned char* host_img = stbi_load(argv[1], &width, &height, &channels, 0);

    //Ensure image is valid
    if(host_img == NULL) {
        perror("Error loading image!\n");
        return EXIT_FAILURE;
    }

    printf("\nImage size:\nwidth: %dpx, height: %dpx, channels: %d\n", width, height, channels);

    int kernel_size;

    //Take kernel size 
    kernel_size = atoi(argv[3]);

    //Ensure valid kernel size
    if(kernel_size % 2 == 0){

        perror("Kernel size must be odd!\n");
        return EXIT_FAILURE;
    }

    //size_t mid_size = kernel_size / 2 ;

    //Dynamic allocation to store kernel in square matrix
    double** kernel = (double**)malloc(sizeof(double*) * kernel_size);
    for(size_t i = 0; i < kernel_size; ++i) 
        kernel[i] = (double*)malloc(sizeof(double)*kernel_size);
    
    generate_kernel(kernel_size, kernel);

    printf("\nGaussian Kernel used for blurring: \n");
    for(size_t i = 0; i < kernel_size; ++i) {

        for(size_t j = 0; j < kernel_size; ++j) 
            printf("%f ", kernel[i][j]);
        
        printf("\n");
    }

    size_t image_size = width * height * channels;

    //Allocating space in host for blurred imaged
    unsigned char* host_blurred_img = (unsigned char*)malloc(image_size);

    unsigned char* dev_img;
    unsigned char* dev_blurred_img;

    cudaError_t err = cudaSuccess;
    //Allocating space in device for image and blurred image
    err = cudaMalloc((void**)&dev_img, image_size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate image vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err =cudaMalloc((void**)&dev_blurred_img, image_size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate blurred image vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Transport image info into device
    err = cudaMemcpy(dev_img, host_img, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy fom host image vector to device image vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    //Allocating space for kernel
    double** dev_kernel;
    err = cudaMalloc((void**)&dev_kernel, sizeof(double*)*kernel_size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate kernel matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double* dev_kernel_rows[kernel_size];
    
    for(int i=0; i<kernel_size; i++){
        err = cudaMalloc((void**)&dev_kernel_rows[i], sizeof(double)*kernel_size);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate a row of kernel matrix (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(dev_kernel_rows[i], kernel[i], sizeof(double)*kernel_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to copy one row from host kernel to device kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
    }

    err = cudaMemcpy(dev_kernel, dev_kernel_rows, sizeof(double*)*kernel_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy from host kernel to device kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    //Get number of blocks and threads per block
    int BLOCKS_PER_GRID = atoi(argv[4]);
    int THREADS_PER_BLOCK = atoi(argv[5]);

    //Total number of threads
    int TOTAL_THREADS = BLOCKS_PER_GRID*THREADS_PER_BLOCK;
    //Calculate number of pixels that each thread is going to operate
    int n_pixels = (width*height)/(TOTAL_THREADS);  

                                                      
    //Call kernel
    blurImage<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_img, dev_blurred_img, dev_kernel,
                                                      kernel_size, n_pixels, TOTAL_THREADS, width, height);
    
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch blurImage kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Get blurred image from device
    err = cudaMemcpy(host_blurred_img, dev_blurred_img, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy from device blurred image to host blurred image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Write blurred image in jpg format
    stbi_write_jpg(argv[2], width, height, 3, host_blurred_img, 100);

    //Free global memory in device
    err = cudaFree(dev_img);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device image vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(dev_blurred_img);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device blurred image vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<kernel_size; i++){
        err = cudaFree(dev_kernel_rows[i]);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to free one row of device kernel matrix (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }      
    
    err = cudaFree(dev_kernel);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Free memory in host 
    for(size_t i = 0; i < kernel_size; ++i) 
        free(kernel[i]);
    
    free(kernel);
    stbi_image_free(host_img);
    free(host_blurred_img);

    err = cudaDeviceReset();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nConvolution correctly done!\n");

    return EXIT_SUCCESS;
}
/*
    MDBNet data pre-processing adaped to 360 degrees images
    Author: June,2024
    Based on ideas from : https://gitlab.com/UnBVision/edgenet360/-/tree/master/src?ref_type=heads 
    and https://github.com/shurans/sscnet
*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

typedef high_resolution_clock::time_point clock_tick;
#define MIN(X, Y) (((X) <= (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) >= (Y)) ? (X) : (Y))

// Voxel information
float vox_unit = 0.02;
float vox_margin = 0.24;
int vox_size_x = 240;
int vox_size_y = 144;
int vox_size_z = 240;

// Camera information
float f = 518.85;
float sensor_w = 640;
float sensor_h = 480;

// GPU parameters
int NUM_THREADS = 1024;
int DEVICE = 0;
int debug = 0;
float cam_back_ = 1.0;
float cam_height_ = 1.0;

// GPU Variables
float* parameters_GPU;
#define VOX_UNIT (0)
#define VOX_MARGIN (1)
#define VOX_SIZE_X (2)
#define VOX_SIZE_Y (3)
#define VOX_SIZE_Z (4)
#define CAM_F (5)
#define SENSOR_W (6)
#define SENSOR_H (7)
#define GO_BACK (8)
#define CAM_HEIGHT (9)

#define FLOOR_OFFSET (0.00)
//VOX_LIMITS
#define OUT_OF_FOV (4)
#define OUT_OF_ROOM (3)
#define OCCLUDED (2)
#define OCCUPIED (1)
#define EMPTY_VISIBLE (0)

#define NUM_CLASSES (256)
#define MAX_DOWN_SIZE (1000)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

clock_tick start_timer() {
    return (high_resolution_clock::now());
}

void end_timer(clock_tick t1, const char msg[]) {
    if (debug == 1) {
        clock_tick t2 = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1).count();
        printf("%s: %ld(ms)\n", msg, duration);
    }
}

//float cam_K[9] = {518.8579f, 0.0f, (float)frame_width / 2.0f, 0.0f, 518.8579f, (float)frame_height / 2.0f, 0.0f, 0.0f, 1.0f};




void setup_CPP(int device, int num_threads, float v_unit, float v_margin,
    float focal_length, float s_w, float s_h,
    int vox_x, int vox_y, int vox_z,
    int debug_flag,
    ///////////////////////////////////////////////////////////////////////////////////
    float cam_height, float cam_back
    ///////////////////////////////////////////////////////////////////////////////////
) {
    DEVICE = device;
    NUM_THREADS = num_threads;
    vox_unit = v_unit;
    vox_margin = v_margin;
    f = focal_length;
    sensor_w = s_w;
    sensor_h = s_h;
    vox_size_x = vox_x;
    vox_size_y = vox_y;
    vox_size_z = vox_z;
    cam_height_ = cam_height;
    cam_back_ = cam_back;
    
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, DEVICE);
    cudaSetDevice(DEVICE);

    if (debug_flag == 1) {

        printf("\nUsing GPU: %s - (device %d)\n", deviceProperties.name, DEVICE);
        printf("Total Memory: %ld\n", deviceProperties.totalGlobalMem);
        printf("Max threads per block: %d\n", deviceProperties.maxThreadsPerBlock);
        printf("Max threads dimension: (%d, %d, %d)\n", deviceProperties.maxGridSize[0],
            deviceProperties.maxGridSize[1],
            deviceProperties.maxGridSize[2]);
        printf("Major, Minor: (%d, %d)\n", deviceProperties.major, deviceProperties.minor);
        printf("Multiprocessor count: %d\n", deviceProperties.multiProcessorCount);
        printf("Threads per block: %d\n", NUM_THREADS);
    }

    debug = debug_flag;

    if (NUM_THREADS > deviceProperties.maxThreadsPerBlock) {
        printf("Selected NUM_THREADS (%d) is greater than device's max threads per block (%d)\n",
            NUM_THREADS, deviceProperties.maxThreadsPerBlock);
        exit(0);
    }


    float parameters[10];

    cudaMalloc(&parameters_GPU, 10 * sizeof(float));

    parameters[VOX_UNIT] = vox_unit;
    parameters[VOX_MARGIN] = vox_margin;
    parameters[CAM_F] = f;
    parameters[SENSOR_W] = sensor_w;
    parameters[SENSOR_H] = sensor_h;
    parameters[VOX_SIZE_X] = (float)vox_size_x;
    parameters[VOX_SIZE_Y] = (float)vox_size_y;
    parameters[VOX_SIZE_Z] = (float)vox_size_z;
    parameters[CAM_HEIGHT] = cam_height_;
    parameters[GO_BACK] = cam_back_;
    cudaMemcpy(parameters_GPU, parameters, 10 * sizeof(float), cudaMemcpyHostToDevice);


}

void clear_parameters_GPU() {
    cudaFree(parameters_GPU);
}


__global__
void point_cloud_kernel(float* baseline, unsigned char* depth_data,
    float* point_cloud, int* width, int* height) {

  //Rerieve pixel coodinates
    int pixel_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (pixel_idx >= (*width * *height))
        return;

    int pixel_y = pixel_idx / *width;
    int pixel_x = pixel_idx % *width;

    float     CV_PI = 3.141592;

    int		max_radius = 30;
    int		inf_border = 160;		// Range (in pixel) from the pole to exclude from point cloud generation
    double	unit_h, unit_w;	//angular size of 1 pixel
    float		disp_scale = 2;
    float		disp_offset = -120;

    unit_h = 1.0 / (*height);
    unit_w = 2.0 / (*width);

    // Get point in world coordinate
    int point_disparity = depth_data[pixel_y * *width + pixel_x];


    float longitude, latitude, radius, angle_disp;

    latitude = pixel_y * unit_h * CV_PI;

    longitude = pixel_x * unit_w * CV_PI;

    point_cloud[6 * pixel_idx + 3] = latitude;
    point_cloud[6 * pixel_idx + 4] = longitude;

    if (point_disparity == 0)
        return;

    if (pixel_y<inf_border || pixel_y> *height - inf_border)
        return;

    angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;

    if (latitude + angle_disp < 0)
        angle_disp = 0.01;

    if (angle_disp == 0) {
        radius = max_radius;
        point_disparity = 0;
    }
    else
        radius = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));

    if (radius > max_radius || radius < 0.0) {
        radius = max_radius;
        point_disparity = 0;
    }

    // spherical to cartesian coordinate
    float rx = radius * sin(latitude) * cos(CV_PI - longitude);
    float rz = radius * sin(latitude) * sin(CV_PI - longitude);
    float ry = radius * cos(latitude); 

    
    point_cloud[6 * pixel_idx + 0] = rx;
    point_cloud[6 * pixel_idx + 1] = ry;
    point_cloud[6 * pixel_idx + 2] = rz;
    point_cloud[6 * pixel_idx + 5] = radius;
    
}


void get_point_cloud_CPP(float baseline, unsigned char* depth_data, float* point_cloud, int width, int height) {

    clock_tick t1 = start_timer();

    float* baseline_GPU;
    int* width_GPU;
    int* height_GPU;
    unsigned char* depth_data_GPU;
    float* point_cloud_GPU;
    int num_pixels = width * height;
    

    gpuErrchk(cudaMalloc(&baseline_GPU, sizeof(float)));
    gpuErrchk(cudaMalloc(&width_GPU, sizeof(int)));
    gpuErrchk(cudaMalloc(&height_GPU, sizeof(int)));
    
    gpuErrchk(cudaMalloc(&depth_data_GPU, num_pixels * sizeof(unsigned char)));
    gpuErrchk(cudaMalloc(&point_cloud_GPU, 6 * num_pixels * sizeof(float)));
    gpuErrchk(cudaMemset(point_cloud_GPU, 0, 6 * num_pixels * sizeof(float)));
    

    gpuErrchk(cudaMemcpy(baseline_GPU, &baseline, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(width_GPU, &width, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(height_GPU, &height, sizeof(int), cudaMemcpyHostToDevice));
    

    gpuErrchk(cudaMemcpy(depth_data_GPU, depth_data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice));

    end_timer(t1, "Prepare duration");

    if (debug == 1) printf("frame width: %d   frame heigth: %d   num_pixels %d\n", width, height, num_pixels);


    t1 = start_timer();
    
    int NUM_BLOCKS = int((width * height + size_t(NUM_THREADS) - 1) / NUM_THREADS);

    if (debug == 1) printf("NUM_BLOCKS: %d   NUM_THREADS: %d\n", NUM_BLOCKS, NUM_THREADS);

    point_cloud_kernel << <NUM_BLOCKS, NUM_THREADS >> > (baseline_GPU, depth_data_GPU, point_cloud_GPU,
        width_GPU, height_GPU);
    
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaDeviceSynchronize());

    end_timer(t1, "depth2Grid duration");

    cudaMemcpy(point_cloud, point_cloud_GPU, 6 * num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(baseline_GPU);
    cudaFree(width_GPU);
    cudaFree(height_GPU);
    cudaFree(depth_data_GPU);
    cudaFree(point_cloud_GPU);
    
    end_timer(t1, "closeup duration");

}



__global__
void get_voxels_kernel(unsigned char* depth_data_GPU, float* point_cloud_GPU, int* point_cloud_size_GPU, 
    float* boundaries_GPU, int* vol_number_GPU, unsigned char* vox_grid_GPU, float* parameters_GPU) {
    
    float vox_unit_GPU = parameters_GPU[VOX_UNIT];
    float sensor_w_GPU = parameters_GPU[SENSOR_W];
    float sensor_h_GPU = parameters_GPU[SENSOR_H];
    float f_GPU = parameters_GPU[CAM_F];
    int vox_size_x_GPU = (int)parameters_GPU[VOX_SIZE_X];
    int vox_size_y_GPU = (int)parameters_GPU[VOX_SIZE_Y];
    int vox_size_z_GPU = (int)parameters_GPU[VOX_SIZE_Z];
    
    float cam_hight_GPU = parameters_GPU[CAM_HEIGHT];
    float go_back_GPU = parameters_GPU[GO_BACK];
    //Rerieve pixel coodinates
    int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (point_idx >= *point_cloud_size_GPU)
        return;

    int x_idx = point_idx * 6 + 0;
    int y_idx = point_idx * 6 + 1;
    int z_idx = point_idx * 6 + 2;
    int lat_idx = point_idx * 6 + 3;
    int long_idx = point_idx * 6 + 4;
    int rd_idx = point_idx * 6 + 5;

    float  min_x = boundaries_GPU[0];
    float  max_x = boundaries_GPU[1];
    float  min_y = boundaries_GPU[2];
    float  max_y = boundaries_GPU[3];
    float  min_z = boundaries_GPU[4];
    float  max_z = boundaries_GPU[5];

    float wx = point_cloud_GPU[x_idx];
    float wy = point_cloud_GPU[y_idx];
    float wz = point_cloud_GPU[z_idx];
    float latitude = point_cloud_GPU[lat_idx];
    float longitude = point_cloud_GPU[long_idx];
    float rd = point_cloud_GPU[rd_idx];

    if ((wx == 0.) && (wy == 0.) && (wz == 0.)) {   
        return;
    }


    if ((wx < min_x) || (wx > max_x) || (wy < min_y) || (wy > max_y) || (wz < min_z) || (wz > max_z)) {
        return;
    }


    int vx, vy, vz;
    
    //Adjust to vol_number
    if (*vol_number_GPU == 1) { // left view ==> rotate 90
        

        float rotated_wx = wz; // Swap wx and wz to simulate a 90-degree rotation
        float rotated_wz = -wx;
        
        float angle_dev_h = (abs(rotated_wz) + abs(rd)) * (sensor_w_GPU / 2) / f_GPU;
        float angle_dev_v = (abs(rotated_wz) + abs(rd)) * (sensor_h_GPU / 2) / f_GPU;
        

        if ((abs(rotated_wx) > angle_dev_h) || (abs(wy - min_y - cam_hight_GPU) > angle_dev_v)) {
            return;
        }

        vz = (int)floor(rotated_wz / vox_unit_GPU);
        vx = (int)floor(rotated_wx / vox_unit_GPU + vox_size_z_GPU / 2);
        vy = (int)floor((wy - (min_y - FLOOR_OFFSET)) / vox_unit_GPU);

    }

    if (*vol_number_GPU == 2) { // front
       
        //Calculating viewing angle deviation for each point
        float angle_dev_h = (abs(wz) + abs(rd)) * (sensor_w_GPU / 2) / f_GPU;
        float angle_dev_v = (abs(wz) + abs(rd)) * (sensor_h_GPU / 2) / f_GPU;
        

        if ((abs(wx) > angle_dev_h) || (abs(wy - min_y - cam_hight_GPU) > angle_dev_v)) {

            return;
        }

        //voxel coordinates
        vz = (int)floor(wz / vox_unit_GPU);
        vx = (int)floor(wx / vox_unit_GPU + vox_size_z_GPU / 2);
        vy = (int)floor((wy - (min_y - FLOOR_OFFSET)) / vox_unit_GPU);

    }

    if (*vol_number_GPU == 3) { // right view

        float rotated_wx = -wz; // Swap wx and wz to simulate the rotation
        float rotated_wz = wx;
        
        
        float angle_dev_h = (abs(rotated_wz) + abs(rd)) * (sensor_w_GPU / 2) / f_GPU;
        float angle_dev_v = (abs(rotated_wz) + abs(rd)) * (sensor_h_GPU / 2) / f_GPU;
        

        if ((abs(rotated_wx) > angle_dev_h) || (abs(wy - min_y - cam_hight_GPU) > angle_dev_v)) {
            return;
        }

        vz = (int)floor(rotated_wz / vox_unit_GPU);
        vx = (int)floor(rotated_wx / vox_unit_GPU + vox_size_z_GPU / 2);
        vy = (int)floor((wy - (min_y - FLOOR_OFFSET)) / vox_unit_GPU);

    }

    if (*vol_number_GPU == 4) { // Back view 
        
        float angle_dev_h = (abs(-wz) + abs(rd)) * (sensor_w_GPU / 2) / f_GPU;
        float angle_dev_v = (abs(-wz) + abs(rd)) * (sensor_h_GPU / 2) / f_GPU;
        

        if ((abs(-wx) > angle_dev_h) || (abs(wy - min_y - cam_hight_GPU) > angle_dev_v)) {
            return;
        }

        vz = (int)floor(-wz / vox_unit_GPU);
        vx = (int)floor(-wx / vox_unit_GPU + vox_size_z_GPU / 2);
        vy = (int)floor((wy - (min_y - FLOOR_OFFSET)) / vox_unit_GPU);

    }
    if (vx >= 0 && vx < vox_size_x_GPU && vy >= 0 && vy < vox_size_y_GPU && vz >= 0 && vz < vox_size_z_GPU) {
        int vox_idx = vz * vox_size_x_GPU * vox_size_y_GPU + vy * vox_size_x_GPU + vx;
        vox_grid_GPU[vox_idx] = float(1.0);
    }

}

__global__
void get_depth_mapping_idx_kernel(unsigned char* vox_grid_GPU, float* depth_mapping_3d_GPU, float* parameters_GPU)
{
    // conver from image corrdinate (point_depth) --> camera coordinate (point_cam) --> world coordinate (point_base)
    float sensor_w_GPU = parameters_GPU[SENSOR_W];
    float sensor_h_GPU = parameters_GPU[SENSOR_H];
    int vox_size_x_GPU = (int)parameters_GPU[VOX_SIZE_X];
    int vox_size_y_GPU = (int)parameters_GPU[VOX_SIZE_Y];
    int vox_size_z_GPU = (int)parameters_GPU[VOX_SIZE_Z];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= 0 && x < vox_size_x_GPU && y >= 0 && y < vox_size_y_GPU && z >= 0 && z < vox_size_z_GPU) {
        int vox_idx = z * vox_size_x_GPU * vox_size_y_GPU + y * vox_size_x_GPU + x;
        if (vox_grid_GPU[vox_idx] > 0 ) {
            int pixel_x = x; // assuming x maps directly to pixel_x
            int pixel_y = y; // assuming y maps directly to pixel_y
            if (pixel_x < int(sensor_w_GPU)&& pixel_y < int(sensor_h_GPU)) {
                depth_mapping_3d_GPU[pixel_y * int(sensor_w_GPU) + pixel_x] = vox_idx;
            }
        }
    }
    
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void get_voxels_CPP(
    /////////////////////
    unsigned char* depth_data,
    /////////////////////
    float* point_cloud, int width, int height, float* boundaries, int vol_number,
    unsigned char* vox_grid,
/////////////////////
float* depth_mapping
////////////////////
) {

    clock_tick t1 = start_timer();

    int point_cloud_size = width * height;

    float* point_cloud_GPU;
    int* point_cloud_size_GPU;
    float* boundaries_GPU;
    int* vol_number_GPU;
    unsigned char* vox_grid_GPU;
    /////////////////////////////////////////
    float* depth_mapping_3d_GPU;
    unsigned char* depth_data_GPU;
    ////////////////////////////////////////
    
    int num_voxels = vox_size_x * vox_size_y * vox_size_z;

    if (debug == 1) printf("get_voxels - point_cloud_size: %d   vol_number: %d  voxel_size: %d %d %d\n",
        point_cloud_size, vol_number, vox_size_x, vox_size_y, vox_size_z);

    if (debug == 1) printf("get_voxels - boundaries: (%2.2f %2.2f) (%2.2f %2.2f) (%2.2f %2.2f)\n",
        boundaries[0], boundaries[1], boundaries[2], boundaries[3], boundaries[4], boundaries[5]);
    
   
    gpuErrchk(cudaMalloc(&point_cloud_GPU, point_cloud_size * 6 * sizeof(float)));
    gpuErrchk(cudaMalloc(&point_cloud_size_GPU, sizeof(int)));
    gpuErrchk(cudaMalloc(&boundaries_GPU, 6 * sizeof(float)));
    gpuErrchk(cudaMalloc(&vol_number_GPU, sizeof(int)));
    gpuErrchk(cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char)));
    /////////////////////////////////////////////////////////////////////
    gpuErrchk(cudaMalloc(&depth_data_GPU, point_cloud_size * sizeof(unsigned char)));
    gpuErrchk(cudaMalloc(&depth_mapping_3d_GPU, int(sensor_h) * int(sensor_w) * sizeof(float)));
    /////////////////////////////////////////////////////////////////////
    
    gpuErrchk(cudaMemcpy(point_cloud_GPU, point_cloud, point_cloud_size * 6 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(point_cloud_size_GPU, &point_cloud_size, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(boundaries_GPU, boundaries, 6 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(vol_number_GPU, &vol_number, sizeof(int), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMemset(vox_grid_GPU, 0, num_voxels * sizeof(unsigned char)));
    
    gpuErrchk(cudaMemcpy(depth_mapping_3d_GPU, depth_mapping, int(sensor_h) * int(sensor_w) * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(depth_data_GPU, depth_data, point_cloud_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    //////////////////////////////////////////////////////////////////
    end_timer(t1, "Prepare duration");

    t1 = start_timer();
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////// test kernal///////////////////////////////////////////////////
    int NUM_BLOCKS = int((point_cloud_size + size_t(NUM_THREADS) - 1) / NUM_THREADS);
  
    if (debug == 1) printf("get_voxels - NUM_BLOCKS: %d   NUM_THREADS: %d\n", NUM_BLOCKS, NUM_THREADS);
    
    
    get_voxels_kernel << <NUM_BLOCKS, NUM_THREADS >> > (depth_data_GPU, point_cloud_GPU, point_cloud_size_GPU, boundaries_GPU, vol_number_GPU, vox_grid_GPU, parameters_GPU);
    
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((vox_size_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (vox_size_y + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (vox_size_z + threadsPerBlock.z - 1) / threadsPerBlock.z);
    get_depth_mapping_idx_kernel << <numBlocks, threadsPerBlock >> > (vox_grid_GPU, depth_mapping_3d_GPU, parameters_GPU);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    end_timer(t1, "get_voxels duration");

  
    
    cudaMemcpy(vox_grid, vox_grid_GPU, num_voxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
    /// //////////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(depth_mapping, depth_mapping_3d_GPU, int(sensor_h) * int(sensor_w) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(depth_data, depth_data_GPU, point_cloud_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    /// //////////////////////////////////////////////////////////////////////////////////////////

    cudaFree(point_cloud_GPU);
    cudaFree(point_cloud_size_GPU);
    cudaFree(boundaries_GPU);
    cudaFree(vol_number_GPU);
    cudaFree(vox_grid_GPU);
    
    ////////////////////////////////////////
    cudaFree(depth_mapping_3d_GPU);
    cudaFree(depth_data_GPU);
    ////////////////////////////////////////
    end_timer(t1, "cleanup duration");

}

__global__
void get_one_hot_kernel(float* point_cloud_GPU, int* point_cloud_size_GPU,
    float* boundaries_GPU, int* one_hot_GPU, float* parameters_GPU) {

    //Rerieve pixel coodinates
    int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_idx >= *point_cloud_size_GPU)
        return;



    float wx = point_cloud_GPU[point_idx * 4 + 0];
    float wy = point_cloud_GPU[point_idx * 4 + 1];
    float wz = point_cloud_GPU[point_idx * 4 + 2];
    float lbl = point_cloud_GPU[point_idx * 4 + 3];

    float  min_x = boundaries_GPU[0];
    float  max_x = boundaries_GPU[1];
    float  min_y = boundaries_GPU[2];
    float  max_y = boundaries_GPU[3];
    float  min_z = boundaries_GPU[4];
    float  max_z = boundaries_GPU[5];

    if ((wx < min_x) || (wx > max_x) || (wy < min_y) || (wy > max_y) || (wz < min_z) || (wz > max_z)) {
        
        return;
    }


    float vox_unit_GPU = parameters_GPU[VOX_UNIT] * 4; // downsampling here

    int vox_size_x_GPU = (int)(parameters_GPU[VOX_SIZE_X] / 2);
    int vox_size_y_GPU = (int)(parameters_GPU[VOX_SIZE_Y] / 4);
    int vox_size_z_GPU = (int)(parameters_GPU[VOX_SIZE_Z] / 2);

    int vx = (int)floor(wx / vox_unit_GPU) + vox_size_x_GPU / 2;
    int vy = (int)floor((wy - min_y) / vox_unit_GPU);
    int vz = (int)floor(wz / vox_unit_GPU) + vox_size_z_GPU / 2;

    if (vx >= 0 && vx < vox_size_x_GPU && vy >= 0 && vy < vox_size_y_GPU && vz >= 0 && vz < vox_size_z_GPU) {
        int vox_idx = vz * vox_size_x_GPU * vox_size_y_GPU * 12 + vy * vox_size_x_GPU * 12 + vx * 12 + lbl;

        atomicAdd(&one_hot_GPU[vox_idx], 1);

    }
    else {
       
        return;
    }
}


__global__
void get_gt_kernel(int* one_hot_GPU, unsigned char* gt_grid_GPU, float* parameters_GPU) {

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int vox_size_x_GPU = (int)(parameters_GPU[VOX_SIZE_X] / 2);
    int vox_size_y_GPU = (int)(parameters_GPU[VOX_SIZE_Y] / 4);
    int vox_size_z_GPU = (int)(parameters_GPU[VOX_SIZE_Z] / 2);

    if (vox_idx >= vox_size_x_GPU * vox_size_y_GPU * vox_size_z_GPU) {
        return;
    }


    int z = (vox_idx / (vox_size_x_GPU * vox_size_y_GPU)) % vox_size_z_GPU;
    int y = (vox_idx / vox_size_x_GPU) % vox_size_y_GPU;
    int x = vox_idx % vox_size_x_GPU;


    int lbl_count = 0;
    int occup_count = 0;
    unsigned char best_lbl = 0;

    for (int i = 0; i < 12; i++) {
        occup_count += one_hot_GPU[vox_idx * 12 + i];
        if (one_hot_GPU[vox_idx * 12 + i] > lbl_count) {
            lbl_count = one_hot_GPU[vox_idx * 12 + i];
            best_lbl = i;
        }
        
    }
    
    if (occup_count > 32) { //reduce noise
        gt_grid_GPU[vox_idx] = best_lbl;
    }

}



void get_gt_CPP(float* point_cloud, int point_cloud_size, float* boundaries, unsigned char* gt_grid) {

    clock_tick t1 = start_timer();

    float* point_cloud_GPU;
    int* point_cloud_size_GPU;
    float* boundaries_GPU;
    int* one_hot_grid_GPU;
    unsigned char* gt_grid_GPU;

    int num_voxels = vox_size_x / 2 * vox_size_y / 4 * vox_size_z / 2;

    if (debug == 1) printf("get_gt - point_cloud_size: %d   voxel_size: %d %d %d\n",
        point_cloud_size, vox_size_x / 2, vox_size_y / 4, vox_size_z / 2);

    if (debug == 1) printf("get_gt - boundaries: (%2.2f %2.2f) (%2.2f %2.2f) (%2.2f %2.2f)\n",
        boundaries[0], boundaries[1], boundaries[2], boundaries[3], boundaries[4], boundaries[5]);

    if (debug == 1) printf("(x %2.2f  y %2.2f z %2.2f l %2.2f\n",
        point_cloud[0], point_cloud[1], point_cloud[2], point_cloud[3]);



    gpuErrchk(cudaMalloc(&point_cloud_GPU, point_cloud_size * 4 * sizeof(float)));
    gpuErrchk(cudaMalloc(&point_cloud_size_GPU, sizeof(int)));
    gpuErrchk(cudaMalloc(&boundaries_GPU, 6 * sizeof(float)));
    gpuErrchk(cudaMalloc(&one_hot_grid_GPU, num_voxels * 12 * sizeof(int)));
    gpuErrchk(cudaMalloc(&gt_grid_GPU, num_voxels * sizeof(unsigned char)));

    gpuErrchk(cudaMemcpy(point_cloud_GPU, point_cloud, point_cloud_size * 4 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(point_cloud_size_GPU, &point_cloud_size, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(boundaries_GPU, boundaries, 6 * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(gt_grid_GPU, 0, num_voxels * sizeof(unsigned char)));
    gpuErrchk(cudaMemset(one_hot_grid_GPU, 0, num_voxels * 12 * sizeof(int)));


    end_timer(t1, "Prepare duration");

    t1 = start_timer();
    int NUM_BLOCKS = int((point_cloud_size + size_t(NUM_THREADS) - 1) / NUM_THREADS);

    if (debug == 1) printf("get_one_hot - NUM_BLOCKS: %d   NUM_THREADS: %d\n", NUM_BLOCKS, NUM_THREADS);

    get_one_hot_kernel << <NUM_BLOCKS, NUM_THREADS >> > (point_cloud_GPU, point_cloud_size_GPU,
        boundaries_GPU, one_hot_grid_GPU, parameters_GPU);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    NUM_BLOCKS = int((num_voxels + size_t(NUM_THREADS) - 1) / NUM_THREADS);

    if (debug == 1) printf("get_gt - NUM_BLOCKS: %d   NUM_THREADS: %d\n", NUM_BLOCKS, NUM_THREADS);

    get_gt_kernel << <NUM_BLOCKS, NUM_THREADS >> > (one_hot_grid_GPU, gt_grid_GPU, parameters_GPU);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    end_timer(t1, "get_gt duration");

    cudaMemcpy(gt_grid, gt_grid_GPU, num_voxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(point_cloud_GPU);
    cudaFree(point_cloud_size_GPU);
    cudaFree(boundaries_GPU);
    cudaFree(one_hot_grid_GPU);
    cudaFree(gt_grid_GPU);

    end_timer(t1, "cleanup duration");

}


__global__
void downsample_grid_kernel(unsigned char* in_grid_GPU, unsigned char* out_grid_GPU, float* parameters_GPU) {

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
    float downscale = 4;

    int in_vox_size_x = (int)parameters_GPU[VOX_SIZE_X];
    int in_vox_size_y = (int)parameters_GPU[VOX_SIZE_Y];
    int in_vox_size_z = (int)parameters_GPU[VOX_SIZE_Z];
    int out_vox_size_x = (int)in_vox_size_x / downscale;
    int out_vox_size_y = (int)in_vox_size_y / downscale;
    int out_vox_size_z = (int)in_vox_size_z / downscale;

    if (vox_idx >= out_vox_size_x * out_vox_size_y * out_vox_size_z) {
        return;
    }

    int z = (vox_idx / (out_vox_size_x * out_vox_size_y)) % out_vox_size_z;
    int y = (vox_idx / out_vox_size_x) % out_vox_size_y;
    int x = vox_idx % out_vox_size_x;

    int sum_occupied = 0;

    for (int tmp_x = x * downscale; tmp_x < (x + 1) * downscale; ++tmp_x) {
        for (int tmp_y = y * downscale; tmp_y < (y + 1) * downscale; ++tmp_y) {
            for (int tmp_z = z * downscale; tmp_z < (z + 1) * downscale; ++tmp_z) {

                int tmp_vox_idx = tmp_z * in_vox_size_x * in_vox_size_y + tmp_y * in_vox_size_z + tmp_x;

                if (in_grid_GPU[tmp_vox_idx] > 0) {
                    sum_occupied += 1;
                }
            }
        }
    }
    if (sum_occupied >= 4) {  //empty threshold
        out_grid_GPU[vox_idx] = 1;
    }
    else {
        out_grid_GPU[vox_idx] = 0;
    }

}

__global__
void downsample_limits_kernel(unsigned char* in_grid_GPU, unsigned char* out_grid_GPU, float* parameters_GPU) {

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
    float downscale = 4;

    int in_vox_size_x = (int)parameters_GPU[VOX_SIZE_X];
    int in_vox_size_y = (int)parameters_GPU[VOX_SIZE_Y];
    int in_vox_size_z = (int)parameters_GPU[VOX_SIZE_Z];
    int out_vox_size_x = (int)in_vox_size_x / downscale;
    int out_vox_size_y = (int)in_vox_size_y / downscale;
    int out_vox_size_z = (int)in_vox_size_z / downscale;

    if (vox_idx >= out_vox_size_x * out_vox_size_y * out_vox_size_z) {
        return;
    }

    int z = (vox_idx / (out_vox_size_x * out_vox_size_y)) % out_vox_size_z;
    int y = (vox_idx / out_vox_size_x) % out_vox_size_y;
    int x = vox_idx % out_vox_size_x;

    int sum_occupied = 0;
    int sum_occluded = 0;

    for (int tmp_x = x * downscale; tmp_x < (x + 1) * downscale; ++tmp_x) {
        for (int tmp_y = y * downscale; tmp_y < (y + 1) * downscale; ++tmp_y) {
            for (int tmp_z = z * downscale; tmp_z < (z + 1) * downscale; ++tmp_z) {

                int tmp_vox_idx = tmp_z * in_vox_size_x * in_vox_size_y + tmp_y * in_vox_size_z + tmp_x;

                if (in_grid_GPU[tmp_vox_idx] == OCCUPIED) {
                    sum_occupied += 1;
                }
                if (in_grid_GPU[tmp_vox_idx] == OCCLUDED) {
                    sum_occluded += 1;
                }
            }
        }
    }
    if (sum_occupied + sum_occluded >= 4) {  //empty threshold
        out_grid_GPU[vox_idx] = 1;
    }
    else {
        out_grid_GPU[vox_idx] = 0;
    }

}


void downsample_grid_CPP(unsigned char* vox_grid, unsigned char* vox_grid_down) {

    clock_tick t1 = start_timer();

    unsigned char* vox_grid_GPU;
    unsigned char* vox_grid_down_GPU;

    int num_voxels = vox_size_x * vox_size_y * vox_size_z;
    int num_voxels_down = num_voxels / 64;

    gpuErrchk(cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char)));
    gpuErrchk(cudaMalloc(&vox_grid_down_GPU, num_voxels_down * sizeof(unsigned char)));

    gpuErrchk(cudaMemcpy(vox_grid_GPU, vox_grid, num_voxels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(vox_grid_down_GPU, 0, num_voxels_down * sizeof(unsigned char)));


    end_timer(t1, "Prepare duration");

    t1 = start_timer();
    int NUM_BLOCKS = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

    if (debug == 1) printf("downsample - NUM_BLOCKS: %d   NUM_THREADS: %d\n", NUM_BLOCKS, NUM_THREADS);

    downsample_grid_kernel << <NUM_BLOCKS, NUM_THREADS >> > (vox_grid_GPU, vox_grid_down_GPU, parameters_GPU);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    end_timer(t1, "downsample duration");

    cudaMemcpy(vox_grid_down, vox_grid_down_GPU, num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(vox_grid_GPU);
    cudaFree(vox_grid_down_GPU);

    end_timer(t1, "cleanup duration");

}


void downsample_limits_CPP(unsigned char* vox_grid, unsigned char* vox_grid_down) {

    clock_tick t1 = start_timer();

    unsigned char* vox_grid_GPU;
    unsigned char* vox_grid_down_GPU;

    int num_voxels = vox_size_x * vox_size_y * vox_size_z;
    int num_voxels_down = num_voxels / 64;

    gpuErrchk(cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char)));
    gpuErrchk(cudaMalloc(&vox_grid_down_GPU, num_voxels_down * sizeof(unsigned char)));

    gpuErrchk(cudaMemcpy(vox_grid_GPU, vox_grid, num_voxels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(vox_grid_down_GPU, 0, num_voxels_down * sizeof(unsigned char)));


    end_timer(t1, "Prepare duration");

    t1 = start_timer();
    int NUM_BLOCKS = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

    if (debug == 1) printf("downsample - NUM_BLOCKS: %d   NUM_THREADS: %d\n", NUM_BLOCKS, NUM_THREADS);

    downsample_limits_kernel << <NUM_BLOCKS, NUM_THREADS >> > (vox_grid_GPU, vox_grid_down_GPU, parameters_GPU);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    end_timer(t1, "downsample duration");

    cudaMemcpy(vox_grid_down, vox_grid_down_GPU, num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(vox_grid_GPU);
    cudaFree(vox_grid_down_GPU);

    end_timer(t1, "cleanup duration");

}

__global__

void SquaredDistanceTransform(unsigned char* depth_data, unsigned char* vox_grid,
    float* vox_tsdf, unsigned char* vox_limits, float* baseline,
    int* width, int* height, float* boundaries_GPU, int* vol_number_GPU, float* parameters_GPU) {

    float vox_unit_GPU = parameters_GPU[VOX_UNIT];
    float vox_margin_GPU = parameters_GPU[VOX_MARGIN];
    float sensor_w_GPU = parameters_GPU[SENSOR_W];
    float sensor_h_GPU = parameters_GPU[SENSOR_H];
    float f_GPU = parameters_GPU[CAM_F];
    float cam_hight_GPU = parameters_GPU[CAM_HEIGHT];
    float go_back_GPU = parameters_GPU[GO_BACK];
    
    int vox_size_x_GPU = (int)parameters_GPU[VOX_SIZE_X];
    int vox_size_y_GPU = (int)parameters_GPU[VOX_SIZE_Y];
    int vox_size_z_GPU = (int)parameters_GPU[VOX_SIZE_Z];

    float  min_x = boundaries_GPU[0];
    float  max_x = boundaries_GPU[1];
    float  min_y = boundaries_GPU[2];
    float  max_y = boundaries_GPU[3];
    float  min_z = boundaries_GPU[4];
    float  max_z = boundaries_GPU[5];

    //Rerieve pixel coodinates
    int search_region = (int)roundf(vox_margin_GPU / vox_unit_GPU);

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (vox_idx >= vox_size_x_GPU * vox_size_y_GPU * vox_size_z_GPU) {
        return;
    }
    
 
    int z = (vox_idx / (vox_size_x_GPU * vox_size_y_GPU)) % vox_size_z_GPU;
    int y = (vox_idx / vox_size_x_GPU) % vox_size_y_GPU;
    int x = vox_idx % vox_size_x_GPU;

    // Get point in world coordinates
    float wz;
    float wx;
    float wy;
    
    //Adjust to vol_number
    if (*vol_number_GPU == 1) { // left view
      
      wx = - (float(z)*vox_unit_GPU);
      wz = (float(x) - (vox_size_x_GPU / 2)) *vox_unit_GPU ;
      wy = float(y) * vox_unit_GPU + (min_y-FLOOR_OFFSET);
      
      if (wx < min_x || wx > max_x || wy < min_y || wy > max_y || wz < min_z || wz > max_z ){
        // outside ROOM
        vox_tsdf[vox_idx] = 2000.;
        vox_limits[vox_idx] = OUT_OF_ROOM;
        return;
      }
      //////////////////////////////////////////////////////////////////////////////////
      float CV_PI = 3.141592;
      float longitude, latitude, point_depth, angle_disp;

      float hip1 = sqrtf(wx * wx + wz * wz);
      float hip2 = sqrtf(hip1 * hip1 + wy * wy);

      float teta1, teta2;

      teta1 = asin(wy / hip2);

      latitude = CV_PI / 2 - teta1;

      if (wx < 0)
          teta2 = asin(wz / hip1);
      else
          teta2 = CV_PI - asin(wz / hip1);

      longitude = teta2;

      float  	unit_h, unit_w;	//angular size of 1 pixel
      float		disp_scale = 2;
      float		disp_offset = -120;
      int		max_radius = 30;


      unit_h = 1.0 / (*height);
      unit_w = 2.0 / (*width);

      int pixel_y = latitude / (unit_h * CV_PI);
      int pixel_x = longitude / (unit_w * CV_PI);

    
          int point_disparity = depth_data[pixel_y * *width + pixel_x];
          if (point_disparity == 0) { // mising depth
              vox_tsdf[vox_idx] = -1.0;
              return;
          }
  
          angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;
  
          if (latitude + angle_disp < 0)
              angle_disp = 0.01;
  
          if (angle_disp == 0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
          else
              point_depth = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));
  
          if (point_depth > max_radius || point_depth < 0.0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
      //}

      float vox_depth = hip2;
      /////////////////////////////////////////////////////////////////////////////////
      
      float angle_dev_h = (abs(-wx) + abs(point_depth)) * (sensor_w_GPU / 2) / f_GPU; //rotated_wx = wz;, rotated_wz = -wx
      float angle_dev_v = (abs(-wx) + abs(point_depth)) * (sensor_h_GPU / 2) / f_GPU;
      
      if ((abs(wz) > angle_dev_h) || (abs(wy - min_y - cam_hight_GPU) > angle_dev_v)) {
        vox_tsdf[vox_idx] = 2000.;
        vox_limits[vox_idx] = OUT_OF_FOV;
        return;
      }
      /////////////////////////////////////////////////////////////////////////////////////////
      if (wx == 0.0 && wy == 0 && wz == 0) {
          vox_tsdf[vox_idx] = 2000.;
          vox_limits[vox_idx] = OUT_OF_FOV;
          return;
      }
      //OCCUPIED
      if (vox_grid[vox_idx] > 0) {
          vox_tsdf[vox_idx] = 0;
          vox_limits[vox_idx] = OCCUPIED;
          return;
      }


      float sign;
      if (abs(point_depth - vox_depth) < 0.001) {
          sign = 1; // avoid NaN
      }
      else {
          sign = (point_depth - vox_depth) / abs(point_depth - vox_depth);
      }
      vox_tsdf[vox_idx] = sign;
      if (sign > 0.0) {
          vox_limits[vox_idx] = EMPTY_VISIBLE;
      }
      else {
          vox_limits[vox_idx] = OCCLUDED;

      }

      //compute the minimum TSDF value for the current voxel within the search region, considering the occupied voxels in the grid
      for (int iix = max(0, x - search_region); iix < min((int)vox_size_x_GPU, x + search_region + 1); iix++) {
          for (int iiy = max(0, y - search_region); iiy < min((int)vox_size_y_GPU, y + search_region + 1); iiy++) {
              for (int iiz = max(0, z - search_region); iiz < min((int)vox_size_z_GPU, z + search_region + 1); iiz++) {

                  int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                  if (vox_grid[iidx] > 0) {

                      float xd = abs(x - iix);
                      float yd = abs(y - iiy);
                      float zd = abs(z - iiz);
                      float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_region;
                      if (tsdf_value < abs(vox_tsdf[vox_idx])) {
                          vox_tsdf[vox_idx] = float(tsdf_value * sign);
                      }
                  }
              }
          }
      }
      //////////////////////////////////////////////////////////////////////////////////////////
      }
      
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
    if (*vol_number_GPU == 2) { // front
      
      wz = float(z) * vox_unit_GPU;                    //point_base[0]
      wx = (float(x) - (vox_size_z_GPU/2)) * vox_unit_GPU; //point_base[1]
      wy = float(y) * vox_unit_GPU + (min_y-FLOOR_OFFSET);            //point_base[2]
      
      if (wx < min_x || wx > max_x || wy < min_y || wy > max_y || wz < min_z || wz > max_z ){
        // outside ROOM
        vox_tsdf[vox_idx] = 2000.;
        vox_limits[vox_idx] = OUT_OF_ROOM;
        return;
      }
      //////////////////////////////////////////////////////////////////////////////////
      float CV_PI = 3.141592;
      float longitude, latitude, point_depth, angle_disp;

      float hip1 = sqrtf(wx * wx + wz * wz);
      float hip2 = sqrtf(hip1 * hip1 + wy * wy);

      float teta1, teta2;

      teta1 = asin(wy / hip2);

      latitude = CV_PI / 2 - teta1;

      if (wx < 0)
          teta2 = asin(wz / hip1);
      else
          teta2 = CV_PI - asin(wz / hip1);

      longitude = teta2;

      float  	unit_h, unit_w;	//angular size of 1 pixel
      float		disp_scale = 2;
      float		disp_offset = -120;
      int		max_radius = 30;


      unit_h = 1.0 / (*height);
      unit_w = 2.0 / (*width);

      int pixel_y = latitude / (unit_h * CV_PI);
      int pixel_x = longitude / (unit_w * CV_PI);

          int point_disparity = depth_data[pixel_y * *width + pixel_x];
          if (point_disparity == 0) { // mising depth
              vox_tsdf[vox_idx] = -1.0;
              return;
          }
  
          angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;
  
          if (latitude + angle_disp < 0)
              angle_disp = 0.01;
  
          if (angle_disp == 0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
          else
              point_depth = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));
  
          if (point_depth > max_radius || point_depth < 0.0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
        //}

      float vox_depth = hip2;
      /////////////////////////////////////////////////////////////////////////////////
      
      float angle_dev_h = (abs(wz) + abs(point_depth)) * (sensor_w_GPU / 2) / f_GPU;
      float angle_dev_v = (abs(wz) + abs(point_depth)) * (sensor_h_GPU / 2) / f_GPU;

      if ((abs(wx)>angle_dev_h) || (abs(wy - min_y - cam_hight_GPU)>angle_dev_v)) {
        vox_tsdf[vox_idx] = 2000.;
        vox_limits[vox_idx] = OUT_OF_FOV;
        return;
      }
      /////////////////////////////////////////////////////////////////////////////////////////
      if (wx == 0.0 && wy == 0 && wz == 0) {
          vox_tsdf[vox_idx] = 2000.;
          vox_limits[vox_idx] = OUT_OF_FOV;
          return;
      }
      //OCCUPIED
      if (vox_grid[vox_idx] > 0) {
          vox_tsdf[vox_idx] = 0;
          vox_limits[vox_idx] = OCCUPIED;
          return;
      }


      float sign;
      if (abs(point_depth - vox_depth) < 0.001) {
          sign = 1; // avoid NaN
      }
      else {
          sign = (point_depth - vox_depth) / abs(point_depth - vox_depth);
      }
      vox_tsdf[vox_idx] = sign;
      if (sign > 0.0) {
          vox_limits[vox_idx] = EMPTY_VISIBLE;
      }
      else {
          vox_limits[vox_idx] = OCCLUDED;

      }

      //compute the minimum TSDF value for the current voxel within the search region, considering the occupied voxels in the grid
      for (int iix = max(0, x - search_region); iix < min((int)vox_size_x_GPU, x + search_region + 1); iix++) {
          for (int iiy = max(0, y - search_region); iiy < min((int)vox_size_y_GPU, y + search_region + 1); iiy++) {
              for (int iiz = max(0, z - search_region); iiz < min((int)vox_size_z_GPU, z + search_region + 1); iiz++) {

                  int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                  if (vox_grid[iidx] > 0) {

                      float xd = abs(x - iix);
                      float yd = abs(y - iiy);
                      float zd = abs(z - iiz);
                      float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_region;
                      if (tsdf_value < abs(vox_tsdf[vox_idx])) {
                          vox_tsdf[vox_idx] = float(tsdf_value * sign);
                      }
                  }
              }
          }
      }
      //////////////////////////////////////////////////////////////////////////////////////////
      }
      
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (*vol_number_GPU == 3) { // right view
        
      wx = (float(z)) * vox_unit_GPU;
      wz = -((float(x) - (vox_size_x_GPU / 2)) * vox_unit_GPU);
      wy = float(y) * vox_unit_GPU + (min_y - FLOOR_OFFSET);

      float rotated_wx = -wz; // Swap wx and wz to simulate a 90-degree rotation
      float rotated_wz = wx;
      //////////////////////////////////////////////////////////////////////////////////
      float CV_PI = 3.141592;
      float longitude, latitude, point_depth, angle_disp;

      float hip1 = sqrtf(wx * wx + wz * wz);
      float hip2 = sqrtf(hip1 * hip1 + wy * wy);

      float teta1, teta2;

      teta1 = asin(wy / hip2);

      latitude = CV_PI / 2 - teta1;

      if (wx < 0)
          teta2 = asin(wz / hip1);
      else
          teta2 = CV_PI - asin(wz / hip1);

      longitude = teta2;

      float  	unit_h, unit_w;	//angular size of 1 pixel
      float		disp_scale = 2;
      float		disp_offset = -120;
      int		max_radius = 30;


      unit_h = 1.0 / (*height);
      unit_w = 2.0 / (*width);

      int pixel_y = latitude / (unit_h * CV_PI);
      int pixel_x = longitude / (unit_w * CV_PI);

          int point_disparity = depth_data[pixel_y * *width + pixel_x];
          if (point_disparity == 0) { // mising depth
              vox_tsdf[vox_idx] = -1.0;
              return;
          }
  
          angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;
  
          if (latitude + angle_disp < 0)
              angle_disp = 0.01;
  
          if (angle_disp == 0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
          else
              point_depth = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));
  
          if (point_depth > max_radius || point_depth < 0.0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
        //}

      float vox_depth = hip2;
      /////////////////////////////////////////////////////////////////////////////////
      float angle_dev_h = (abs(rotated_wz) + abs(point_depth)) * (sensor_w_GPU / 2) / f_GPU;
      float angle_dev_v = (abs(rotated_wz) + abs(point_depth)) * (sensor_h_GPU / 2) / f_GPU;
      

      if ((abs(rotated_wx) > angle_dev_h) || (abs(wy - min_y - cam_hight_GPU) > angle_dev_v)) {
         vox_tsdf[vox_idx] = 2000.;
         vox_limits[vox_idx] = OUT_OF_FOV;
         return;
      }
      /////////////////////////////////////////////////////////////////////////////////////////
      if (wx == 0.0 && wy == 0 && wz == 0) {
          vox_tsdf[vox_idx] = 2000.;
          vox_limits[vox_idx] = OUT_OF_FOV;
          return;
      }
      //OCCUPIED
      if (vox_grid[vox_idx] > 0) {
          vox_tsdf[vox_idx] = 0;
          vox_limits[vox_idx] = OCCUPIED;
          return;
      }


      float sign;
      if (abs(point_depth - vox_depth) < 0.001) {
          sign = 1; // avoid NaN
      }
      else {
          sign = (point_depth - vox_depth) / abs(point_depth - vox_depth);
      }
      vox_tsdf[vox_idx] = sign;
      if (sign > 0.0) {
          vox_limits[vox_idx] = EMPTY_VISIBLE;
      }
      else {
          vox_limits[vox_idx] = OCCLUDED;

      }

      //compute the minimum TSDF value for the current voxel within the search region, considering the occupied voxels in the grid
      for (int iix = max(0, x - search_region); iix < min((int)vox_size_x_GPU, x + search_region + 1); iix++) {
          for (int iiy = max(0, y - search_region); iiy < min((int)vox_size_y_GPU, y + search_region + 1); iiy++) {
              for (int iiz = max(0, z - search_region); iiz < min((int)vox_size_z_GPU, z + search_region + 1); iiz++) {

                  int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                  if (vox_grid[iidx] > 0) {

                      float xd = abs(x - iix);
                      float yd = abs(y - iiy);
                      float zd = abs(z - iiz);
                      float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_region;
                      if (tsdf_value < abs(vox_tsdf[vox_idx])) {
                          vox_tsdf[vox_idx] = float(tsdf_value * sign);
                      }
                  }
              }
          }
      }
      //////////////////////////////////////////////////////////////////////////////////////////
      
      }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
    if (*vol_number_GPU == 4) { // back view  
            
        wz = (float(-z) * vox_unit_GPU);                    //point_base[0]
        wx = -((float(x) - (vox_size_z_GPU / 2)) * vox_unit_GPU); //point_base[1]
        wy = float(y) * vox_unit_GPU + (min_y - FLOOR_OFFSET);            //point_base[2]

        if (wx < min_x || wx > max_x || wy < min_y || wy > max_y || wz < min_z || wz > max_z) {
            // outside ROOM
            vox_tsdf[vox_idx] = 2000.;
            vox_limits[vox_idx] = OUT_OF_ROOM;
            return;
        }
        //////////////////////////////////////////////////////////////////////////////////
        float CV_PI = 3.141592;
        float longitude, latitude, point_depth, angle_disp;

        float hip1 = sqrtf(wx * wx + wz * wz);
        float hip2 = sqrtf(hip1 * hip1 + wy * wy);

        float teta1, teta2;

        teta1 = asin(wy / hip2);

        latitude = CV_PI / 2 - teta1;

        if (wx < 0)
            teta2 = asin(wz / hip1);
        else
            teta2 = CV_PI - asin(wz / hip1);

        longitude = teta2;

        float  	unit_h, unit_w;	//angular size of 1 pixel
        float		disp_scale = 2;
        float		disp_offset = -120;
        int		max_radius = 30;

        unit_h = 1.0 / (*height);
        unit_w = 2.0 / (*width);

        int pixel_y = latitude / (unit_h * CV_PI);
        int pixel_x = longitude / (unit_w * CV_PI);

          int point_disparity = depth_data[pixel_y * *width + pixel_x];
          if (point_disparity == 0) { // mising depth
              vox_tsdf[vox_idx] = -1.0;
              return;
          }
  
          angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;
  
          if (latitude + angle_disp < 0)
              angle_disp = 0.01;
  
          if (angle_disp == 0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
          else
              point_depth = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));
  
          if (point_depth > max_radius || point_depth < 0.0) {
              point_depth = max_radius;
              point_disparity = 0;
          }
       
        float vox_depth = hip2;

        //////////////////////////////////////////////////////////////////////////////////////////
        
        float angle_dev_h = (abs(-wz) + abs(point_depth)) * (sensor_w_GPU / 2) / f_GPU;
        float angle_dev_v = (abs(-wz) + abs(point_depth)) * (sensor_h_GPU / 2) / f_GPU;
        

        if ((abs(-wx) > angle_dev_h) || (abs(wy - min_y - cam_hight_GPU) > angle_dev_v)) {
            vox_tsdf[vox_idx] = 2000.;
            vox_limits[vox_idx] = OUT_OF_FOV;
            return;
        }
        //////////////////////////////////////////////////////////////////////////////////////////
        
        if (wx == 0.0 && wy == 0 && wz == 0) {
            vox_tsdf[vox_idx] = 2000.;
            vox_limits[vox_idx] = OUT_OF_FOV;
            return;
        }
        //OCCUPIED
        if (vox_grid[vox_idx] > 0) {
            vox_tsdf[vox_idx] = 0;
            vox_limits[vox_idx] = OCCUPIED;
            return;
        }


        float sign;
        if (abs(point_depth - vox_depth) < 0.001) {
            sign = 1; // avoid NaN
        }
        else {
            sign = (point_depth - vox_depth) / abs(point_depth - vox_depth);
        }
        vox_tsdf[vox_idx] = sign;
        if (sign > 0.0) {
            vox_limits[vox_idx] = EMPTY_VISIBLE;
        }
        else {
            vox_limits[vox_idx] = OCCLUDED;

        }

        //compute the minimum TSDF value for the current voxel within the search region, considering the occupied voxels in the grid
        for (int iix = max(0, x - search_region); iix < min((int)vox_size_x_GPU, x + search_region + 1); iix++) {
            for (int iiy = max(0, y - search_region); iiy < min((int)vox_size_y_GPU, y + search_region + 1); iiy++) {
                for (int iiz = max(0, z - search_region); iiz < min((int)vox_size_z_GPU, z + search_region + 1); iiz++) {

                    int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                    if (vox_grid[iidx] > 0) {

                        float xd = abs(x - iix);
                        float yd = abs(y - iiy);
                        float zd = abs(z - iiz);
                        float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_region;
                        if (tsdf_value < abs(vox_tsdf[vox_idx])) {
                            vox_tsdf[vox_idx] = float(tsdf_value * sign);
                        }
                    }
                }
            }
        }
        //////////////////////////////////////////////////////////////////////////////////////////
    }
}



void FlipTSDF_CPP(float* vox_tsdf) {

    clock_tick t1 = start_timer();

    for (int vox_idx = 0; vox_idx < vox_size_x * vox_size_y * vox_size_x; vox_idx++) {

        float value = float(vox_tsdf[vox_idx]);
        if (value > 1)
            value = 1;


        float sign;
        if (abs(value) < 0.001)
            sign = 1;
        else
            sign = value / abs(value);

        vox_tsdf[vox_idx] = sign * (max(0.001, (1.0 - abs(value))));
    }
    end_timer(t1, "FlipTSDF");
}


void FTSDFDepth_CPP(unsigned char* depth_data,
    unsigned char* vox_grid,
    float* vox_tsdf,
    unsigned char* vox_limits,
    float baseline,
    int width,
    int height,
    float* boundaries,
    int vol_number) {

    clock_tick t1 = start_timer();

    float* boundaries_GPU;
    unsigned char* vox_grid_GPU;
    
    unsigned char* depth_data_GPU;
    float* vox_tsdf_GPU;
    
    unsigned char* vox_limits_GPU;
    float* baseline_GPU;
    int* width_GPU;
    int* height_GPU;
    int* vol_number_GPU;

    int num_voxels = vox_size_x * vox_size_y * vox_size_z;
    int num_pixels = width * height; // spherical image size

    if (debug == 1) printf("FTSDFDepth - boundaries: (%2.2f %2.2f) (%2.2f %2.2f) (%2.2f %2.2f)\n",
        boundaries[0], boundaries[1], boundaries[2], boundaries[3], boundaries[4], boundaries[5]);
    
    
    gpuErrchk(cudaMalloc(&boundaries_GPU, 6 * sizeof(float)));
    gpuErrchk(cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char)));
  
    gpuErrchk(cudaMalloc(&depth_data_GPU, num_pixels * sizeof(unsigned char)));
    gpuErrchk(cudaMalloc(&vox_tsdf_GPU, num_voxels * sizeof(float)));
    
    gpuErrchk(cudaMalloc(&vox_limits_GPU, num_voxels * sizeof(unsigned char)));
    gpuErrchk(cudaMalloc(&baseline_GPU, sizeof(float)));
    gpuErrchk(cudaMalloc(&width_GPU, sizeof(int)));
    gpuErrchk(cudaMalloc(&height_GPU, sizeof(int)));
    gpuErrchk(cudaMalloc(&vol_number_GPU, sizeof(int)));

    gpuErrchk(cudaMemcpy(boundaries_GPU, boundaries, 6 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(vox_grid_GPU, vox_grid, num_voxels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMemcpy(depth_data_GPU, depth_data, num_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
   
    
    gpuErrchk(cudaMemcpy(baseline_GPU, &baseline, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(width_GPU, &width, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(height_GPU, &height, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(vol_number_GPU, &vol_number, sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(vox_limits_GPU, 0, num_voxels * sizeof(unsigned char)));
    gpuErrchk(cudaMemset(vox_tsdf_GPU, 0, num_voxels * sizeof(float)));
    


    end_timer(t1, "Prepare duration");

    t1 = start_timer();
    int NUM_BLOCKS = int((num_voxels + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  
    SquaredDistanceTransform << <NUM_BLOCKS, NUM_THREADS >> > (depth_data_GPU, vox_grid_GPU, vox_tsdf_GPU, vox_limits_GPU, baseline_GPU, width_GPU, height_GPU, boundaries_GPU, vol_number_GPU, parameters_GPU);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    end_timer(t1, "SquaredDistanceTransform duration");
    cudaMemcpy(vox_tsdf, vox_tsdf_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(vox_limits, vox_limits_GPU, num_voxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(boundaries_GPU);
    cudaFree(vox_grid_GPU);
    cudaFree(depth_data_GPU);
    cudaFree(vox_tsdf_GPU);
    cudaFree(vox_limits_GPU);
    cudaFree(baseline_GPU);
    cudaFree(width_GPU);
    cudaFree(height_GPU);
    cudaFree(vol_number_GPU);

    end_timer(t1, "cleanup duration");

    FlipTSDF_CPP(vox_tsdf);
  
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" {
    void get_point_cloud(float baseline,
        unsigned char* depth_data,
        float* point_cloud,
        int width,
        int height
        ) {
        get_point_cloud_CPP(baseline,
            depth_data,
            point_cloud,
            width,
            height
            );
    }
    


    void get_voxels(
        /// ////////////////////////////////////////////////////////////////////////////////
        unsigned char* depth_data, 
        /// ////////////////////////////////////////////////////////////////////////////////
        float* point_cloud,
        int width, int height,
        float* boundaries,
        int vol_number,
        unsigned char* vox_grid,
       
        /// ////////////////////////////////////////////////////////////////////////////////
        float* depth_mapping_3d
        /// ////////////////////////////////////////////////////////////////////////////////
    ) {
        get_voxels_CPP(
            /// ////////////////////////////////////////////////////////////////////////////////
            depth_data, 
            /// ////////////////////////////////////////////////////////////////////////////////
            point_cloud,
            width, height,
            boundaries,
            vol_number,
            vox_grid,
            /// ////////////////////////////////////////////////////////////////////////////////
            depth_mapping_3d
            /// ////////////////////////////////////////////////////////////////////////////////
        );
    }

    void get_gt(float* point_cloud,
        int point_cloud_size,
        float* boundaries,
        unsigned char* gt_grid) {
        get_gt_CPP(point_cloud,
            point_cloud_size,
            boundaries,
            gt_grid);
    }

    void FTSDFDepth(unsigned char* depth_data,
        unsigned char* vox_grid,
        float* vox_tsdf,
        unsigned char* vox_limits,
        float baseline,
        int width,
        int height,
        float* boundaries,
        int vol_number) {
        FTSDFDepth_CPP(depth_data,
            vox_grid,
            vox_tsdf,
            vox_limits,
            baseline,
            width,
            height,
            boundaries,
            vol_number);
    }
    
    
    void downsample_grid(unsigned char* vox_grid,
        unsigned char* vox_grid_down) {
        downsample_grid_CPP(vox_grid,
            vox_grid_down);
    }
    void downsample_limits(unsigned char* vox_grid,
        unsigned char* vox_grid_down) {
        downsample_limits_CPP(vox_grid,
            vox_grid_down);
    }
    void setup(int device, int num_threads,
        float v_unit, float v_margin,
        float f, float sensor_w, float sensor_h,
        int vox_size_x, int vox_size_y, int vox_size_z,
        int debug_flag,
        ///////////////////////////////////////////////////////
        float cam_height,
        float cam_back
        ) {
        setup_CPP(device, num_threads,
            v_unit, v_margin,
            f, sensor_w, sensor_h,
            vox_size_x, vox_size_y, vox_size_z,
            debug_flag,
            ///////////////////////////////////////////////////////   
            cam_height,
            cam_back);
    }


    void finish() {
        clear_parameters_GPU();
    }
}


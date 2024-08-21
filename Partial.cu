#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__device__ float sigma_function(float x) {
    // Replace this with the actual sigma function if different
    return x; // Identity function for now
}

__global__ void divide_kernel(float *W, float *Q, int num_rows, int num_cols, int c, float beta, float gamma, int g, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < c) {
        float g_power = powf(g, row);
        float k_power = powf(k, col);

        // Numerator
        float numerator = W[row * num_cols + col] + beta * g_power * k_power + gamma;

        // Denominator
        float sigma_value = sigma_function(g_power * k_power);
        float denominator = W[row * num_cols + col] + beta * sigma_value + gamma;

        // Compute Q
        Q[row * c + col] = (denominator != 0) ? (numerator / denominator) : 0; // Prevent division by zero
    }
}


__global__ void reduce_kernel(float *Q, float *QQ, int num_rows, int c, int f_q) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < c / f_q) {
        float tmp = 1.0f;
        for (int l = 0; l < f_q; l++) {
            tmp *= Q[row * c + col * f_q + l];
        }
        QQ[row * (c / f_q) + col] = tmp;
    }
}

__global__ void prefix_product_kernel(float *QQ, float *P, int total_size) {
    __shared__ float shared_mem[1024]; // Adjust size as needed

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        shared_mem[threadIdx.x] = (idx == 0) ? 1 : QQ[idx - 1];
        __syncthreads();

        if (idx > 0) {
            P[idx] = P[idx - 1] * shared_mem[threadIdx.x];
        } else {
            P[idx] = shared_mem[threadIdx.x];
        }
    }
}


void printMatrix(const char* label, float* matrix, int rows, int cols) {
    std::cout << label << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Matrix dimensions and parameters
    int d = 2; // log2 of number of rows
    int num_rows = 1 << d;  // Number of rows in witness
    int m = 5;  // Number of columns in witness
    int c = 4;  // Number of columns in output

    // Parameters
    int g = 2;
    int k = 7;
    float beta = 4;
    float gamma = 5;
    int f_q = 2; // Reduction factor

    // Witness matrix initialization
    float W[num_rows * m] = {0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 13, 14, 15, 5, 17, 18, 19}; 
    W[2] = W[1]; 
    W[12] = W[5];
    W[16] = W[5];

    // Allocate result arrays
    float Q[num_rows * c];
    float QQ[num_rows * (c / f_q)];
    float P[num_rows * (c / f_q)];

    // Allocate device memory
    float *d_W, *d_Q, *d_QQ, *d_P;
    cudaMalloc((void **)&d_W, num_rows * m * sizeof(float));
    cudaMalloc((void **)&d_Q, num_rows * c * sizeof(float));
    cudaMalloc((void **)&d_QQ, num_rows * (c / f_q) * sizeof(float));
    cudaMalloc((void **)&d_P, num_rows * (c / f_q) * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_W, W, num_rows * m * sizeof(float), cudaMemcpyHostToDevice);

    // Print witness matrix on host
    printMatrix("W", W, num_rows, m);

    // Step 1: Divide
    dim3 blockSize(16, 16);
    dim3 gridSize((c + blockSize.x - 1) / blockSize.x, (num_rows + blockSize.y - 1) / blockSize.y);
    divide_kernel<<<gridSize, blockSize>>>(d_W, d_Q, num_rows, m, c, beta, gamma, g, k);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Step 2: Reduce
    reduce_kernel<<<gridSize, blockSize>>>(d_Q, d_QQ, num_rows, c, f_q);

    // Check for CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Step 3: Prefix Product
    int total_size = num_rows * (c / f_q);
    prefix_product_kernel<<<(total_size + blockSize.x - 1) / blockSize.x, blockSize.x>>>(d_QQ, d_P, total_size);

    // Check for CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(Q, d_Q, num_rows * c * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(QQ, d_QQ, num_rows * (c / f_q) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(P, d_P, num_rows * (c / f_q) * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printMatrix("Q (after Divide)", Q, num_rows, c);
    printMatrix("Q' (after Reduce)", QQ, num_rows, c / f_q);
    printMatrix("P (after Prefix Product)", P, num_rows, c / f_q);

    // Free device memory
    cudaFree(d_W);
    cudaFree(d_Q);
    cudaFree(d_QQ);
    cudaFree(d_P);

    return 0;
}

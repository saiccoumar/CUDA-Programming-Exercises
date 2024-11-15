#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 5000
__global__ void matrixAdd(const float *A, const float *B, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int index = row * N + col;
        C[index] = A[index] + B[index];
    }
}

void printMatrix(const float *matrix)
{
    if (N <= 5)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                std::cout << matrix[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    } else{
        std::cout << "Matrix too large." << std::endl; 
    }
}

void matrixAddCPU(const float *A, const float *B, float *C)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}

int main()
{
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];
    float *d_A, *d_B, *d_C;

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i)
    {
        A[i] = static_cast<float>(i + 1);
    }

    for (int i = 0; i < N * N; ++i)
    {
        B[i] = static_cast<float>(N * N - 1 - i);
    }

    std::cout << "Matrix A:" << std::endl;
    printMatrix(A);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(B);

    // Allocate memory on GPU
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA kernel parameters
    dim3 threadsPerBlock(1000, 1000);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Record CUDA kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch CUDA kernel
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Stop and compute CUDA execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix C (A + B) using CUDA:" << std::endl;
    printMatrix(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Measure CPU execution time
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixAddCPU(A, B, C);
    auto stopCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(stopCPU - startCPU).count();
    std::cout << "CPU execution time: " << durationCPU * 1e-6 << " ms" << std::endl;

    // Print CPU result
    std::cout << "Result matrix C (A + B) using CPU:" << std::endl;
    printMatrix(C);

    // Free heap memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

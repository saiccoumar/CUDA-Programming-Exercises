#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define P 500   // Rows in A
#define N 1000  // Columns in A and rows in B
#define Q 600   // Columns in B

__global__ void matrixMultiply(const float *A, const float *B, float *C, int p, int n, int q)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            // Correct formula for matrix multiplication
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void printMatrix(const float *matrix, int rows, int cols)
{
    if (rows <= 5 && cols <= 5)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "Matrix too large to print." << std::endl;
    }
}

void matrixMultiplyCPU(const float *A, const float *B, float *C, int p, int n, int q)
{
    for (int i = 0; i < p; ++i)
    {
        for (int j = 0; j < q; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * q + j];
            }
            C[i * q + j] = sum;
        }
    }
}

int main()
{
    // Allocate host memory for A, B, and C
    float *A = new float[P * N];
    float *B = new float[N * Q];
    float *C = new float[P * Q];
    float *d_A, *d_B, *d_C;

    // Initialize matrices A and B
    for (int i = 0; i < P * N; ++i)
    {
        A[i] = static_cast<float>(i + 1);
    }

    for (int i = 0; i < N * Q; ++i)
    {
        B[i] = static_cast<float>(N * Q - 1 - i);
    }

    std::cout << "Matrix A:" << std::endl;
    printMatrix(A, P, N);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(B, N, Q);

    // Allocate memory on GPU
    cudaMalloc((void **)&d_A, P * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * Q * sizeof(float));
    cudaMalloc((void **)&d_C, P * Q * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, A, P * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * Q * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA kernel parameters
    dim3 threadsPerBlock(N, N);  // 16x16 thread blocks
    dim3 numBlocks((Q + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (P + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Record CUDA kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch CUDA kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, P, N, Q);
    cudaDeviceSynchronize();

    // Stop and compute CUDA execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(C, d_C, P * Q * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix C (A x B) using CUDA:" << std::endl;
    printMatrix(C, P, Q);

    cudaDeviceSynchronize();
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Measure CPU execution time
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C, P, N, Q);
    auto stopCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(stopCPU - startCPU).count();
    std::cout << "CPU execution time: " << durationCPU * 1e-6 << " ms" << std::endl;

    // Print CPU result
    std::cout << "Result matrix C (A x B) using CPU:" << std::endl;
    printMatrix(C, P, Q);

    // Free heap memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

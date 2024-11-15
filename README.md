# GPU-Programming-Exercises
Brief collection of GPU exercises (my reimplementation). Comes with relevant resources.

### Writing CUDA code
While there are a few ways to write CUDA code, the most robust and popular I've seen is writing C++ cude code and compiling it with NVCC. You can write CUDA code with C++ syntax, but must certain CUDA specific functions included in <cuda_runtime.h> such as cudaMemcpy and cudaFree. More resources on how to write CUDA code to efficiently CUDA cores can be found below:

https://youtu.be/G-EimI4q-TQ?feature=shared

https://youtu.be/kUqkOAU84bA?feature=shared

https://youtu.be/xwbD6fL5qC8?feature=shared

https://www.nvidia.com/content/cudazone/download/Exercise_instructions.pdf

https://github.com/csc-training/CUDA/blob/master/exercises/README.md

### Hardware audits
To check your NVIDIA GPU's information, run
```
nvidia-smi
```
To check the GPU version, run
```
nvcc --version
```

### Compiling CUDA code
First ensure that CUDA and the nvcc compiler are compiled. Then compile a program with the following command:
```
nvcc -o my_program my_program.cu
```
and run your program with the following:
```
./my_program
```

### Evaluating CUDA code
NVIDIA's tools for CUDA code aren't immediately intuitive but provide very detailed information about your code. First, you can record the runtime of a CUDA function without transfer overheads factored within the code and print it out to stdout:
```
...
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
...
```

Additionally, there are a few monitoring tools you can use to observe your CUDA performance. For example, you can use NVIDIA nsight systems to look at the execution timeline of your code on the GPU:
![image](https://github.com/user-attachments/assets/27a44358-34e9-4473-abca-879f3761a34f)

NVIDIA tools change rapidly, so there's no guarantee that this will still be available as a tool in the near future, nor that it will be the best tool available. 

### Matrix Multiplication Example:
Matrix Multiplication is a quintessential parallelized GPU task. I implemented it in matrix_multiplication.cu and compiled it for windows. Here are the performance metrics:
```
CPU (Intel i13700k) Execution Time: 471.213 ms
4070ti Execution Time: 0.424288 ms
2080ti Execution Time: 0.224864 ms
```

Note that a given GPU can have a margin of 0.3 ms margin of runtime which is why a 2080ti appears to run faster than a 4070ti. The difference between GPU's is less noticeable as most GPUs, even older ones, can handle a simple task like matrix multiplication on small (ish) scales well.

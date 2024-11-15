#include <cstdio>
#include <cmath>

__global__ void hello()
{
  printf("Hello\n");
}


int main(void)
{
  int count, device;
  cudaGetDeviceCount(&count);
  cudaGetDevice(&device);
  printf("You have in total %d devices in your system\n", count);
  printf("GPU %d will now print a message for you:\n", device);

  hello<<<2,2>>>();
  cudaDeviceSynchronize();
  
  return 0;	
}
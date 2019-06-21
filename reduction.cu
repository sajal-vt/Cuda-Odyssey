#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;
//using namespace std::chrono;

int test_reduce(int* v);

using namespace std;


__global__ void reduce0(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
		sdata[tid] += sdata[tid + s];
	}
	__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int test_reduce(int* in, int N) {
   
    

	int* d_in;
	int* d_out;
    
        int num_threads = 32;
	int num_blocks = N / num_threads;
	int *out = new int[num_blocks];
	
	cudaMalloc(&d_in, N * sizeof(int));
        cudaMalloc(&d_out, num_blocks * sizeof(int));

        cudaMemcpy(d_in, in, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, out, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

        reduce0<<<num_blocks, num_threads, num_threads * sizeof(int)>>>(d_in, d_out);

        int res = 0;

        cudaMemcpy(out, d_out, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);

	for(int i = 0; i < num_blocks; i++)
	{
            //std::cout << out[i] << std::endl;
	    res += out[i];
	}
        cudaFree(d_in);
        cudaFree(d_out);
        //delete in;
        delete out;
	
	

    return res;
}



int main() {
    int N = 1024;
    int* in = new int[N];
    for (int i = 0; i < N; i++) {
        in[i] = i + 1;
    }

    int maximo = 0;
    for (int i = 0; i < N; i++) {
        maximo += in[i];// std::max(maximo, vec[i]);
    }
    cout << "Max CPU " << maximo << endl;

    int max_cuda = test_reduce(in, N);

    cout << "Max GPU " << max_cuda << endl;
    delete in;
    return 0;
}

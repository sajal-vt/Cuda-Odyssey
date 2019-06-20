#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <random>
#include <utility>
#include <algorithm>
#include <iostream>

using namespace std;
using namespace std::chrono;

int test_reduce(std::vector<int> &v);

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

int test_reduce(std::vector<int> &v) {
    int *in;
    

	int* d_in;
	int* d_out;
    
    int num_threads = 32;
	int num_blocks = v.size() / num_threads;
	int *out = new int[num_blocks];
	
	cudaMalloc(&d_in, v.size() * sizeof(int));
    cudaMalloc(&d_out, num_blocks * sizeof(int));

    cudaMemcpy(d_in, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(out, 0, num_blocks * sizeof(int));

    reduce0<<<num_blocks, num_threads>>>(d_in, d_out);

    int res = 0;

    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < num_blocks; i++)
	{
		res += out[i];
	}
    cudaFree(d_in);
    cudaFree(d_out);
	free(in);
	free(out);
	
	

    return res;
}



int main() {
    int N = 1024;
    std::vector<int> vec(N);

    std::random_device dev;
    std::mt19937 mt(dev());
    std::uniform_int_distribution<> dist(0, N);

    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = dist(mt);
    }

    int maximo = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        maximo = std::max(maximo, vec[i]);// std::max(maximo, vec[i]);
    }
    cout << "Max CPU " << maximo << endl;

    int max_cuda = test_reduce(vec);

    cout << "Max GPU " << max_cuda << endl;

    return 0;
}
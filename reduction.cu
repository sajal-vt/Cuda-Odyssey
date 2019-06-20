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


// Global max reduce example based on CppCon 2016: “Bringing Clang and C++ to GPUs: An Open-Source, CUDA-Compatible GPU C++ Compiler"
__global__ void d_max_reduce(const int *in, int *out, size_t N) {
    int sum = 0;
    size_t start = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    for (size_t i = start; i < start + 4 && i < N; i++) {
        sum = max(__ldg(in + i), sum);
    }

    for (int i = 16; i; i >>= 1) {
        sum = max(__shfl_down(sum, i), sum);
    }

    __shared__ int shared_sum;
    shared_sum = 0;
    __syncthreads();

    if (threadIdx.x % 32 == 0) {
        atomicMax(&shared_sum, sum);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMax(out, shared_sum);
    }
}

int test_reduce(std::vector<int> &v) {
    int *in;
    int *out;

    cudaMalloc(&in, v.size() * sizeof(int));
    cudaMalloc(&out, sizeof(int));

    cudaMemcpy(in, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(out, 0, sizeof(int));

    int threads = 32;

    d_max_reduce<<<1, threads / 4>>>(in, out, v.size());

    int res;

    cudaMemcpy(&res, out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);

    return res;
}



int main() {
    int N = 64;
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
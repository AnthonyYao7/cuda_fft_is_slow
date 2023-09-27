#include <iostream>
#include <cuComplex.h>
#include <random>
#include <complex>
#include <vector>
#include <chrono>

#define MAX_THREADS_PER_BLOCK 1024

using namespace std::chrono;

__global__ void s0_butterfly(const cuDoubleComplex *x, cuDoubleComplex *X, unsigned int l2N);
__global__ void s1_butterfly(unsigned int s, cuDoubleComplex *X);
__device__ size_t bit_reverse(size_t x, size_t bits);

void cuda_malloc_with_error(void** ptr, size_t size);
void cuda_memcpy_with_error(void* dst, void* src, size_t cnt, cudaMemcpyKind kind);
void cuda_free_with_error(void* ptr);

unsigned int min_val(unsigned int a, unsigned int b);
unsigned int max_val(unsigned int a, unsigned int b);

int l2(int x);
std::complex<double> comp_exp(double th);

bool all_close(const std::vector<std::complex<double>>& a, const cuDoubleComplex* b, size_t N, double threshold);

void fft(const cuDoubleComplex* x_h, int N, cuDoubleComplex* X_h);
std::vector<std::complex<double>> fft_baseline(const std::vector<std::complex<double>>& x);


int main() {
    int N = 1 << 24;

    std::random_device rd;
    std::default_random_engine gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto* x = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * N);
    std::vector<std::complex<double>> baseline;

    for (int i = 0; i < N; ++i)
    {
        double Re = dist(gen);
        double Im = dist(gen);

        x[i] = make_cuDoubleComplex(Re, Im);
        baseline.emplace_back(Re, Im);
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    auto gt = fft_baseline(baseline);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto baseline_time = duration_cast<microseconds>(t2 - t1).count();


    auto* X = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * N);


    t1 = high_resolution_clock::now();

    fft(x, N, X);

    t2 = high_resolution_clock::now();
    auto gpu_time = duration_cast<microseconds>(t2 - t1).count();

    free(x);

    if (not all_close(gt, X, N, 0.001))
    {
        std::cout << "failed" << std::endl;
    }

    std::cout << "CPU FFT took " << baseline_time << " microseconds" << std::endl;
    std::cout << "GPU FFT took " << gpu_time << " microseconds" << std::endl;

    free(X);

    return 0;
}


void fft(const cuDoubleComplex* x_h, int N, cuDoubleComplex* X_h)
{
    cuDoubleComplex* X_d = nullptr, *x_d = nullptr;

    size_t size = N * sizeof(cuDoubleComplex);

    // allocate space for input and output arrays
    cuda_malloc_with_error((void **)&X_d, size);
    cuda_malloc_with_error((void **)&x_d, size);

    // copy input array to device
    cuda_memcpy_with_error((void *)x_d, (void *)x_h, size, cudaMemcpyHostToDevice);

    unsigned int l2n = l2(N);

    // time the fft computation only, without the memory copying from host to device and device to host
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // s == 1
    {
        dim3 loop_vars(MAX_THREADS_PER_BLOCK, 1);
        dim3 blocks(N / 2 / MAX_THREADS_PER_BLOCK, 1);
        s0_butterfly<<<blocks, loop_vars>>>(x_d, X_d, l2n);
    }

    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "Failed to launch s=1 butterfly kernel: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // s > 1
    for (size_t s = 2; s < N; s *= 2) {
        dim3 loop_vars;

        // weird logic to calculate the size of each thread block
        if (N > 2 * MAX_THREADS_PER_BLOCK)
        {
            if (s > (N / 2/ s))
            {
                loop_vars.y = min_val(s, MAX_THREADS_PER_BLOCK);
                loop_vars.x = max_val(MAX_THREADS_PER_BLOCK / loop_vars.y, 1);
            }
            else
            {
                loop_vars.x = min_val(N / 2 / s, MAX_THREADS_PER_BLOCK);
                loop_vars.y = max_val(MAX_THREADS_PER_BLOCK / loop_vars.x, 1);
            }
        }

        dim3 blocks((N / 2 / s) / loop_vars.x, s / loop_vars.y);

        s1_butterfly<<<blocks, loop_vars>>>(s, X_d);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "Failed to launch s>1 butterfly kernel: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto baseline_time = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "GPU FFT computation only (not including memory copies) took " << baseline_time << " microseconds" << std::endl;

    // return result
    cuda_memcpy_with_error(X_h, X_d, size, cudaMemcpyDeviceToHost);

    // free device memory
    cuda_free_with_error(X_d);
    cuda_free_with_error(x_d);
}

__global__ void s0_butterfly(const cuDoubleComplex *x, cuDoubleComplex *X, unsigned int l2N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    cuDoubleComplex f = make_cuDoubleComplex(1., 0.);

    unsigned int i1 = 2 * i;
    unsigned int i2 = 2 * i + 1;
    unsigned int n1 = bit_reverse(i1, l2N);
    unsigned int n2 = bit_reverse(i2, l2N);

    cuDoubleComplex t1 = cuCadd(x[n1], cuCmul(f, x[n2]));
    cuDoubleComplex t2 = cuCsub(x[n1], cuCmul(f, x[n2]));

    X[i1] = t1;
    X[i2] = t2;
}

__global__ void s1_butterfly(unsigned int s, cuDoubleComplex *X)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y * blockDim.y + threadIdx.y;

    cuDoubleComplex f = make_cuDoubleComplex(cos(-M_PI * (double)(c) / (double)(s)), sin(-M_PI * (double)(c) / (double)(s)));

    unsigned int i1 = 2 * i * s + c;
    unsigned int i2 = 2 * i * s + c + s;

    cuDoubleComplex t1 = cuCadd(X[i1], cuCmul(f, X[i2]));
    cuDoubleComplex t2 = cuCsub(X[i1], cuCmul(f, X[i2]));

    X[i1] = t1;
    X[i2] = t2;
}

__device__ size_t bit_reverse(size_t x, size_t bits)
{
    size_t ret=0;

    for (;bits > 0; bits--)
    {
        ret<<=1;
        if (x&1)
            ret++;
        x>>=1;
    }
    return ret;
}

void cuda_malloc_with_error(void** ptr, size_t size)
{
    cudaError_t err = cudaMalloc(ptr, size);

    if (err != cudaSuccess)
    {
        std::cout << "Failed to allocate " << size << " bytes with error code " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cuda_memcpy_with_error(void* dst, void* src, size_t cnt, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, cnt, kind);

    if (err != cudaSuccess)
    {
        std::cout << "Failed to copy vector from " << (kind == cudaMemcpyHostToDevice ? "host to device " : "device to host ") << cudaGetErrorString(err) << std::endl;
        exit( EXIT_FAILURE);
    }
}

void cuda_free_with_error(void* ptr)
{
    cudaError_t err = cudaFree(ptr);

    if (err != cudaSuccess)
    {
        std::cout << "Failed to free memory" << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int l2(int x)
{
    int r = 0;
    while (x > 1)
    {
        ++r;
        x /= 2;
    }
    return r;
}

unsigned int min_val(unsigned int a, unsigned int b)
{
    return a > b ? b : a;
}

unsigned int max_val(unsigned int a, unsigned int b)
{
    return a > b ? a : b;
}

bool all_close(const std::vector<std::complex<double>>& a, const cuDoubleComplex* b, size_t N, double threshold)
{
    for (size_t i = 0; i < min_val(a.size(), N); ++i)
    {
        if (std::abs(a[i].real() - b[i].x) > threshold or std::abs(a[i].imag() - b[i].y) > threshold)
        {
            return false;
        }
    }

    return true;
}

std::complex<double> comp_exp(double th)
{
    return {cos(th), sin(th)};
}

size_t bit_rev(size_t x, size_t bits)
{
    size_t ret=0;
    for (;bits > 0; bits--)
    {
        ret<<=1;
        if (x&1)
            ret++;
        x>>=1;
    }
    return ret;
}


std::vector<std::complex<double>> fft_baseline(const std::vector<std::complex<double>>& x)
{
    std::vector<std::complex<double>> X(x.size(), 0.);
    std::complex<double> t1, t2, f;
    size_t i1, i2, n1, n2;

    for (size_t s = 1; s < x.size(); s*=2)
    {
        for (size_t i = 0; i < x.size() / 2 / s; i++)
        {
            for (size_t c = 0; c < s; c++) {
                f = comp_exp(-M_PI * (double)c / (double)s);
                i1 = 2 * i * s + c;
                i2 = 2 * i * s + c + s;
                if (s == 1)
                {
                    n1 = bit_rev(i1, (size_t) l2((int)x.size()));
                    n2 = bit_rev(i2, (size_t) l2((int)x.size()));
                    t1 = x[n1] + f * x[n2];
                    t2 = x[n1] - f * x[n2];
                    X[i1] = t1;
                    X[i2] = t2;
                }
                else
                {
                    t1 = X[i1] + f * X[i2];
                    t2 = X[i1] - f * X[i2];
                    X[i1] = t1;
                    X[i2] = t2;
                }
            }
        }
    }
    return X;
}

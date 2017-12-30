#ifndef UTILS_HPP
#define UTILS_HPP

#include <string.h>

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

//copied from caffe
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col);//dilation set to 1

void col2im_cpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im);//dilation set to 1

void im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col);/*untested!*///dilation set to 1

void col2im_gpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im);/*untested!*///dilation set to 1

void create_cublas_context();/*untested!*/
void destroy_cublas_context();/*untested!*/

void gemm_cpu(bool transA, bool transB,
              int M, int N, int K,
              float alpha,
              float *A,float *B,
              float beta,
              float *C);

void gemm_gpu(bool transA, bool transB,
              int M, int N, int K,
              float alpha,
              float *A,float *B,
              float beta,
              float *C);/*untested!*/

void axpy_cpu(int N,float alpha, float *X, float *Y);
void axpy_gpu(int N,float alpha, float *X, float *Y);/*untested!*/


//used in calculating loss, temporarily unnecessary.
//void nrm2_cpu();
//void nrm2_gpu();

//void copy_cpu();
//void copy_gpu();


/*
class Blob {
public:
	int c_,h_,w_;
	int count_;
	float *cpu_data;
	float *cpu_diff;
	Blob(int c,int h,int w);
};*/

class Blob {
public:
    int c_,h_,w_;
    int count_;
    float *cpu_data;// = 0;
    float *cpu_diff;// = 0;
    float *gpu_data;// = 0;
    float *gpu_diff;// = 0;
    bool is_gpu_;// = false;
    bool data_only_;// = false;
    Blob(int c,int h,int w,bool is_gpu_blob = false,bool data_only = false);
    void set_gpu_data(float *src);/*untested!*/
    void get_gpu_data(float *dst);/*untested!*/
    void clear_gpu();/*untested!*/
    void clear_gpu(bool data, bool diff);
    void alloc_cpu_data();
    void free_cpu_data();
private:
    void init_cpu_blob();
    void init_gpu_blob();/*untested!*/

};

#endif




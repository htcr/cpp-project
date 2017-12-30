#include "utils.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>


__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

//dilation set to 1
void im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      ((kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      ((kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, 1, 1, height_col,
      width_col, data_col);
  //CUDA_POST_KERNEL_CHECK;
}


__global__ void col2im_gpu_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

//dilation set to 1
void col2im_gpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im) {
  int height_col = (height + 2 * pad_h - ((kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - ((kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, 1, 1,
      height_col, width_col, data_im);
  //CUDA_POST_KERNEL_CHECK;
}

//copied from caffe

cublasHandle_t handle;

void create_cublas_context(){
	cublasCreate(&handle);
}

void destroy_cublas_context(){
	cublasDestroy(handle);
}


void gemm_gpu(bool transA, bool transB,
              int M, int N, int K,
              float alpha,
              float *A,float *B,
              float beta,
              float *C){
	cublasOperation_t TA = transA?CUBLAS_OP_T:CUBLAS_OP_N;
	cublasOperation_t TB = transB?CUBLAS_OP_T:CUBLAS_OP_N;
	int lda = transA?M:K;
	int ldb = transB?K:N;
	cublasSgemm(handle,
				TB,TA,
				N,M,K,
				&alpha,
				B,ldb,
				A,lda,
				&beta,
				C,N);
}

void axpy_gpu(int N,float alpha, float *X, float *Y){
	cublasSaxpy(handle,N,&alpha,X,1,Y,1);
}

//the size of src must equal to count_
void Blob::set_gpu_data(float *src){
	cudaMemcpy(gpu_data,src,sizeof(float)*count_,cudaMemcpyHostToDevice);
}

//the size of dst must equal to count_
void Blob::get_gpu_data(float *dst){
	cudaMemcpy(dst,gpu_data,sizeof(float)*count_,cudaMemcpyDeviceToHost);
}

void Blob::clear_gpu(){
	if(data_only_){
		cudaMemset(gpu_data,0,sizeof(float)*count_);
	}else{
		cudaMemset(gpu_data,0,sizeof(float)*count_);
		cudaMemset(gpu_diff,0,sizeof(float)*count_);
	}
}

void Blob::clear_gpu(bool data,bool diff){
	if(data_only_){
		if(data){
			cudaMemset(gpu_data,0,sizeof(float)*count_);
		}
	}else{
		if(data){
			cudaMemset(gpu_data,0,sizeof(float)*count_);
		}
		if(diff){
			cudaMemset(gpu_diff,0,sizeof(float)*count_);	
		}		
	}
}

float gpu_memory_used = 0;

void Blob::init_gpu_blob(){
	if(data_only_){
		cudaMalloc((void**)&gpu_data,sizeof(float)*count_);
		cudaMemset(gpu_data,0,sizeof(float)*count_);
		gpu_memory_used+=1.0*sizeof(float)*count_;
		printf("total gpu memory used: %f MB\n",gpu_memory_used/1024.0/1024.0);
	}else{
		cudaMalloc((void**)&gpu_data,sizeof(float)*count_);
		cudaMemset(gpu_data,0,sizeof(float)*count_);
		cudaMalloc((void**)&gpu_diff,sizeof(float)*count_);
		cudaMemset(gpu_diff,0,sizeof(float)*count_);
		gpu_memory_used+=2.0*sizeof(float)*count_;
		printf("total gpu memory used: %f MB\n",gpu_memory_used/1024.0/1024.0);
	}
}
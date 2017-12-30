#include <utils.hpp>
#include <stdlib.h>

extern "C"{
  #include <cblas.h>
}

void caffe_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col) {
  const int output_h = (height + 2 * pad_h -
    ((kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    ((kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void col2im_cpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im) {
  caffe_set(height * width * channels, float(0), data_im);
  const int output_h = (height + 2 * pad_h -
    ((kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    ((kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void gemm_cpu(bool transA, bool transB,
              int M, int N, int K,
              float alpha,
              float *A,float *B,
              float beta,
              float *C){
  CBLAS_TRANSPOSE TA = transA?CblasTrans:CblasNoTrans;
  CBLAS_TRANSPOSE TB = transB?CblasTrans:CblasNoTrans;
  int lda = transA?M:K;
  int ldb = transB?K:N;
  cblas_sgemm(CblasRowMajor,
              TA,TB,
              M,N,K,
              alpha,
              A,lda,
              B,ldb,
              beta,
              C,N);
}

void axpy_cpu(int N,float alpha, float *X, float *Y){
  cblas_saxpy(N,alpha,X,1,Y,1);
};

/*
Blob::Blob(int c,int h,int w){
  c_ = c;
  h_ = h;
  w_ = w;
  count_ = c*h*w;
  cpu_data = (float*)malloc(sizeof(float)*count_);
  cpu_diff = (float*)malloc(sizeof(float)*count_);
  memset(cpu_data,0,sizeof(float)*count_);
  memset(cpu_diff,0,sizeof(float)*count_);
}*/

Blob::Blob(int c,int h,int w,bool is_gpu_blob, bool data_only){
  is_gpu_ = is_gpu_blob;
  data_only_ = data_only;
  c_ = c;
  h_ = h;
  w_ = w;
  count_ = c*h*w;
  if(!is_gpu_blob){
    init_cpu_blob();
  }else{
    init_gpu_blob();
  }
}

void Blob::init_cpu_blob(){  
  if(data_only_){
    cpu_data = (float*)malloc(sizeof(float)*count_);
    memset(cpu_data,0,sizeof(float)*count_);
  }else{
    cpu_data = (float*)malloc(sizeof(float)*count_);
    cpu_diff = (float*)malloc(sizeof(float)*count_);
    memset(cpu_data,0,sizeof(float)*count_);
    memset(cpu_diff,0,sizeof(float)*count_);  
  }  
}

void Blob::alloc_cpu_data(){
  cpu_data = (float*)malloc(sizeof(float)*count_);
  memset(cpu_data,0,sizeof(float)*count_);
}

void Blob::free_cpu_data(){
  free(cpu_data);
}
#include <layers.hpp>
#include <utils.hpp>
#include <cuda_runtime.h>

#include <stdio.h>

void ConvLayer::init_gpu_convlayer(int M, int N, int K){
	//these doesn't need gpu_diff field
	col_buff_ = new Blob(1,K,N,true,true);
	kernel_   = new Blob(1,M,K,true,true);
	bias_     = new Blob(1,M,1,true,true);
	bias_multiplier_ = new Blob(1,1,N,true,true);
	bias_multiplier_->alloc_cpu_data();
	for(int i = 0;i<N;i++){
		bias_multiplier_->cpu_data[i] = 1;
	}
	bias_multiplier_->set_gpu_data(bias_multiplier_->cpu_data);
	bias_multiplier_->free_cpu_data();
}

void ConvLayer::forward_gpu(Blob *bottom, Blob *top){
	im2col_gpu(bottom->gpu_data, n_in_,
    in_h_, in_w_, k_size_, k_size_,
    pad_, pad_,
    stride_, stride_,
    col_buff_->gpu_data);

	//printf("in\n");
    gemm_gpu(false,false,
			 M_,N_,K_,
			 1,
			 kernel_->gpu_data,col_buff_->gpu_data,
			 0,
			 top->gpu_data);

    gemm_gpu(false,false,
			 M_,N_,1,
			 1,
			 bias_->gpu_data,bias_multiplier_->gpu_data,
			 1,
			 top->gpu_data);
    //printf("out\n");
}

void ConvLayer::backward_gpu(Blob *in, Blob *out){
	//printf("in\n");
	gemm_gpu(true,false,
			 K_,N_,M_,
			 1,
			 kernel_->gpu_data,out->gpu_diff,
			 0,
			 col_buff_->gpu_data);
	//col2im
	//printf("out\n");
	col2im_gpu(col_buff_->gpu_data, n_in_,
    in_h_, in_w_, k_size_, k_size_,
    pad_, pad_,
    stride_, stride_,
    in->gpu_diff);
}




__global__ void AvePoolForward(const int nthreads,
    const float* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    float aveval = 0;
    const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

void PoolLayer::forward_gpu(Blob *bottom, Blob *top){
	//copied from caffe
	AvePoolForward<<<CAFFE_GET_BLOCKS((top->count_)), CAFFE_CUDA_NUM_THREADS>>>(
        top->count_, bottom->gpu_data, 1, n_in_,
        in_h_, in_w_, out_h_, out_w_, k_size_,
        k_size_, stride_, stride_, 0, 0, top->gpu_data);
}

//copied from caffe
__global__ void AvePoolBackward(const int nthreads, const float* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    float* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    float gradient = 0;
    const float* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

void PoolLayer::backward_gpu(Blob *in, Blob *out){
	AvePoolBackward<<<CAFFE_GET_BLOCKS((out->count_)), CAFFE_CUDA_NUM_THREADS>>>(
        out->count_, out->gpu_diff, 1, n_in_,
        in_h_, in_w_, out_h_, out_w_, k_size_,
        k_size_, stride_, stride_, 0, 0, in->gpu_diff);
}

void PoolLayer::pool_mean_gpu(float *bottom_data, float *top_data){

}

void PoolLayer::pool_mean_back_gpu(float *bottom_data, float *top_data){

}

//copied from caffe
__global__ void ReLUForward(const int n, const float* in, float* out,
    float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

void ReluLayer::forward_gpu(Blob *bottom, Blob *top){
	ReLUForward<<<CAFFE_GET_BLOCKS((count_)), CAFFE_CUDA_NUM_THREADS>>>(
      count_, bottom->gpu_data, top->gpu_data, 0);
}

__global__ void ReLUBackward(const int n, const float* in_diff,
    const float* in_data, float* out_diff, float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

void ReluLayer::backward_gpu(Blob *in, Blob *out){
	ReLUBackward<<<CAFFE_GET_BLOCKS((count_)), CAFFE_CUDA_NUM_THREADS>>>(
        count_, out->gpu_diff, in->gpu_data, in->gpu_diff, 0);
}
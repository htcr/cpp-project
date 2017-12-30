//extern "C"{
	#include <cblas.h>
//}

#include <layers.hpp>
#include <utils.hpp>
#include <stdlib.h>

#include <iostream>
using namespace std;

inline int min(int a, int b) {
	return a > b ? b : a;
}

inline int max(int a, int b) {
	return a > b ? a : b;
}

ConvLayer::ConvLayer(int pad,int k_size,int stride,int in_h,int in_w,int n_in,int n_out,bool is_gpu){
	pad_ = pad;
	k_size_ = k_size;
	stride_ = stride;
	in_h_ = in_h;
	in_w_ = in_w;
	out_h_ = (in_h + 2 * pad - ((k_size - 1) + 1)) / stride + 1;
	out_w_ = (in_w + 2 * pad - ((k_size - 1) + 1)) / stride + 1;
	n_in_ = n_in;
	n_out_ = n_out;
	M_=n_out_;
	N_=out_h_*out_w_;
	K_=n_in_*k_size_*k_size_;
	is_gpu_ = is_gpu;

	if(!is_gpu){
		init_cpu_convlayer(M_,N_,K_);
	}else{
		init_gpu_convlayer(M_,N_,K_);
	}

	/*
	col_buff_ = (float*)malloc(sizeof(float)*(K_)*(N_));
	kernel_ = (float*)malloc(sizeof(float)*(M_)*(K_));
	bias_ = (float*)malloc(sizeof(float)*(M_)*1);
	bias_multiplier_ = (float*)malloc(sizeof(float)*N_);
	for(int i = 0;i<N_;i++){
		bias_multiplier_[i] = 1;
	}
	*/
}

void ConvLayer::init_cpu_convlayer(int M, int N, int K){
	//these doesn't need cpu_diff field
	col_buff_ = new Blob(1,K,N,false,true);
	kernel_   = new Blob(1,M,K,false,true);
	bias_     = new Blob(1,M,1,false,true);
	bias_multiplier_ = new Blob(1,1,N,false,true);
	for(int i = 0;i<N;i++){
		bias_multiplier_->cpu_data[i] = 1;
	}
}

void ConvLayer::forward_cpu(Blob *bottom, Blob *top){

	im2col_cpu(bottom->cpu_data, n_in_,
    in_h_, in_w_, k_size_, k_size_,
    pad_, pad_,
    stride_, stride_,
    col_buff_->cpu_data);

	//conv
	/*
	cblas_sgemm(CblasRowMajor,
	CblasNoTrans,CblasNoTrans,
	M_,N_,K_,
	1,
	kernel_->cpu_data,K_,
	col_buff_->cpu_data,N_,
	0,
	top->cpu_data,N_);
	*/
	gemm_cpu(false,false,
			 M_,N_,K_,
			 1,
			 kernel_->cpu_data,col_buff_->cpu_data,
			 0,
			 top->cpu_data);
	//bias
	/*
	cblas_sgemm(CblasRowMajor,
	CblasNoTrans,CblasNoTrans,
	M_,N_,1,
	1,
	bias_->cpu_data,1,
	bias_multiplier_->cpu_data,N_,
	1,
	top->cpu_data,N_);
	*/
	gemm_cpu(false,false,
			 M_,N_,1,
			 1,
			 bias_->cpu_data,bias_multiplier_->cpu_data,
			 1,
			 top->cpu_data);
	

}

void ConvLayer::backward_cpu(Blob *bottom, Blob *top){
	//backward_conv
	/*
	cblas_sgemm(CblasRowMajor,
	CblasTrans,CblasNoTrans,
	K_,N_,M_,
	1,
	kernel_->cpu_data,K_,
	top->cpu_diff,N_,
	0,
	col_buff_->cpu_data,N_);
	*/
	gemm_cpu(true,false,
			 K_,N_,M_,
			 1,
			 kernel_->cpu_data,top->cpu_diff,
			 0,
			 col_buff_->cpu_data);
	//col2im
	col2im_cpu(col_buff_->cpu_data, n_in_,
    in_h_, in_w_, k_size_, k_size_,
    pad_, pad_,
    stride_, stride_,
    bottom->cpu_diff);
}


ReluLayer::ReluLayer(int in_c,int in_h,int in_w){
	count_ = in_c*in_h*in_w;
}

void ReluLayer::forward_cpu(Blob *bottom, Blob *top){
	
	for(int i = 0;i<count_;i++){
		top->cpu_data[i] = ((bottom->cpu_data[i])>0)?(bottom->cpu_data[i]):(0);
	}
	
}

void ReluLayer::backward_cpu(Blob *bottom, Blob *top){
	
	for(int i = 0;i<count_;i++){
		bottom->cpu_diff[i] = (top->cpu_diff[i])*((bottom->cpu_data[i])>0);
	}
	
}


PoolLayer::PoolLayer(int k_size,int stride,int in_h,int in_w,int n_in){
	n_in_ = n_in;
	in_h_ = in_h;
	in_w_ = in_w;
	stride_ = stride;
	k_size_ = k_size;

	//int pool_size_ = k_size_*k_size_; 
	out_h_ = (in_h_ - ((k_size_ - 1) + 1)) / stride_ + 1;
	out_w_ = (in_w_ - ((k_size_ - 1) + 1)) / stride_ + 1;
	in_offset_ = in_h_*in_w_;
	out_offset_ = out_h_*out_w_;
}

void PoolLayer::forward_cpu(Blob *bottom, Blob *top){
	pool_mean_cpu(bottom->cpu_data,top->cpu_data);
}

void PoolLayer::backward_cpu(Blob *bottom, Blob *top){
	pool_mean_back_cpu(bottom->cpu_diff,top->cpu_diff);
}

void PoolLayer::pool_mean_cpu(float *bottom_data, float *top_data){
	for (int c = 0; c < n_in_; ++c) {
        for (int ph = 0; ph < out_h_; ++ph) {
          for (int pw = 0; pw < out_w_; ++pw) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = min(hstart + k_size_, in_h_);
            int wend = min(wstart + k_size_, in_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, in_h_);
            wend = min(wend, in_w_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * out_w_ + pw] +=
                    bottom_data[h * in_w_ + w];
              }
            }
            top_data[ph * out_w_ + pw] /= pool_size;
          }
        }
        // next channel
        bottom_data += in_offset_;
        top_data += out_offset_;
    }
}

void PoolLayer::pool_mean_back_cpu(float *bottom_diff, float *top_diff){
	for (int c = 0; c < n_in_; ++c) {
        for (int ph = 0; ph < out_h_; ++ph) {
          for (int pw = 0; pw < out_w_; ++pw) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = min(hstart + k_size_, in_h_);
            int wend = min(wstart + k_size_, in_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, in_h_);
            wend = min(wend, in_w_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * in_w_ + w] +=
                  top_diff[ph * out_w_ + pw] / pool_size;
              }
            }
          }
        }
        // next channel
        bottom_diff += in_offset_;
        top_diff += out_offset_;
    }
}
#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <utils.hpp>

class Layer{
public:
	virtual void forward_cpu(Blob *bottom, Blob *top){};
	virtual void backward_cpu(Blob *in, Blob *out){};
	virtual void forward_gpu(Blob *bottom, Blob *top){};
	virtual void backward_gpu(Blob *in, Blob *out){};
	virtual ~Layer(){};
};

class ConvLayer : public Layer{
public:

	//read from proto
	int pad_;
	int k_size_;
	int stride_;

	//get from input
	int in_h_;
	int in_w_;

	//out_h = (height + 2 * pad_h - ((kernel_h - 1) + 1)) / stride_h + 1;
	int out_h_;
	int out_w_;

	//get from input
	int n_in_;
	//read from proto
	int n_out_;

	int M_;//M_=n_out_;
	int N_;//N_=out_h*out_w;
	int K_;//K_==n_in_*k_size_*k_size_;

	//size = K_*N_
	//float *col_buff_;
	Blob *col_buff_;


	//read from proto
	//float *kernel_;
	//float *bias_;
	Blob *kernel_;
	Blob *bias_;

	//float *bias_multiplier_;
	Blob *bias_multiplier_;

	bool is_gpu_;

	ConvLayer(int pad,int k_size,int stride,int in_h,int in_w,int n_in,int n_out, bool is_gpu=false);
	virtual void forward_cpu(Blob *bottom, Blob *top);
	virtual void backward_cpu(Blob *in, Blob *out);
	virtual void forward_gpu(Blob *bottom, Blob *top);/*untested!*/;
	virtual void backward_gpu(Blob *in, Blob *out);/*untested!*/;
private:
	void init_cpu_convlayer(int M, int N, int K);
	void init_gpu_convlayer(int M, int N, int K);/*untested!*/

};

class PoolLayer : public Layer{
public:
	int n_in_;
	int in_h_;
	int in_w_;
	int stride_;
	int k_size_;

	int	out_h_;
	int	out_w_;
	int in_offset_; 
	int out_offset_;

	PoolLayer(int k_size,int stride,int in_h,int in_w,int n_in);
	virtual void forward_cpu(Blob *bottom, Blob *top);
	virtual void backward_cpu(Blob *in, Blob *out);
	virtual void forward_gpu(Blob *bottom, Blob *top);/*untested!*/
	virtual void backward_gpu(Blob *in, Blob *out);/*untested!*/
private:
	void pool_mean_cpu(float *bottom_data, float *top_data);
	void pool_mean_back_cpu(float *bottom_data, float *top_data);
	void pool_mean_gpu(float *bottom_data, float *top_data);/*no need!*/
	void pool_mean_back_gpu(float *bottom_data, float *top_data);/*no need!*/
};

class ReluLayer : public Layer{
public:
	int count_;//count_ = in_c*in_h_*in_w_
	ReluLayer(int in_c,int in_h,int in_w);
	virtual void forward_cpu(Blob *bottom, Blob *top);
	virtual void backward_cpu(Blob *in, Blob *out);
	virtual void forward_gpu(Blob *bottom, Blob *top);/*untested!*/
	virtual void backward_gpu(Blob *in, Blob *out);/*untested!*/

};

#endif
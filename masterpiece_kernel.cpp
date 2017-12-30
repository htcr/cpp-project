#include "masterpiece_kernel.h"
#include "layers.hpp"
#include "utils.hpp"
#include "modelreader.hpp"
#include "loss.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <stdlib.h>
#include <string.h>
using namespace std;


#define layer_num 30
#define style_spots 3
bool use_gpu = true;
int content_id = 11;
int max_iterations = 100;
float content_factor = 0.00001;
float style_factor = 0.5;
float rate_origin = 0.001;
float rate = 0.001;
float momentum = 0.9;
float rate_fold = 0.75;
int count_fold = 100;
int style_after_id[5] = { 1,6,11,20,29 };
float style_weights[5] = { 5000,500,250,10,10 };
bool produce = true;

float *kernel_progress;

float *content, *style;
float mean_content, mean_style;
int c, h, w;
ModelReader *mr;

Layer *layers[layer_num];
Blob  *blobs[layer_num + 1];
Blob  *G[layer_num + 1];
Blob  *A[layer_num + 1] = { NULL };
float stws[layer_num + 1];
Blob  *P;

Blob  *out_buff;
Blob  *input_diff;

int n_layer = style_after_id[style_spots - 1] + 1;
int n_blob = n_layer + 1;

void setup_gpu() {
	for (int i = 0; i<5; i++) {
		stws[style_after_id[i] + 1] = style_weights[i];
	}

	create_cublas_context();
}
void setup_cpu() {
	for (int i = 0; i<5; i++) {
		stws[style_after_id[i] + 1] = style_weights[i];
	}
}

void build_net_gpu() {
	//build net
	//build input blob
	blobs[0] = new Blob(c, h, w, true, false);
	//build layers and activation blobs
	for (int i = 0; i<n_layer; i++) {
		int pad, k_size, stride;
		int in_h, in_w;
		int n_in, n_out;
		int type = mr->get_conv_param(i, &pad, &k_size, &stride, &n_out);
		if (type == 0) {
			cout << "building id: " << i << " conv: " << endl;
			ConvLayer *tmp_cv = new ConvLayer(pad, k_size, stride, blobs[i]->h_, blobs[i]->w_, blobs[i]->c_, n_out, true);
			//load kernel and bias
			mr->get_conv_kb_gpu(i, tmp_cv);
			layers[i] = tmp_cv;
			blobs[i + 1] = new Blob(n_out, tmp_cv->out_h_, tmp_cv->out_w_, true, false);
		}
		else if (type == 1) {
			cout << "building id: " << i << " pool: " << endl;
			PoolLayer *tmp_pl = new PoolLayer(k_size, stride, blobs[i]->h_, blobs[i]->w_, blobs[i]->c_);
			layers[i] = tmp_pl;
			blobs[i + 1] = new Blob(blobs[i]->c_, tmp_pl->out_h_, tmp_pl->out_w_, true, false);

		}
		else if (type == 2) {
			cout << "building id: " << i << " relu: " << endl;
			ReluLayer *tmp_re = new ReluLayer(blobs[i]->c_, blobs[i]->h_, blobs[i]->w_);
			layers[i] = tmp_re;
			//in-place
			blobs[i + 1] = blobs[i];
		}
	}

	out_buff = new Blob(c, h, w);
	input_diff = new Blob(c, h, w, true, true);
}
void build_net_cpu() {
	//build net
	//build input blob
	blobs[0] = new Blob(c, h, w);
	//build layers and activation blobs
	for (int i = 0; i<n_layer; i++) {
		int pad, k_size, stride;
		int in_h, in_w;
		int n_in, n_out;
		int type = mr->get_conv_param(i, &pad, &k_size, &stride, &n_out);
		if (type == 0) {
			cout << "building id: " << i << " conv: " << endl;
			ConvLayer *tmp_cv = new ConvLayer(pad, k_size, stride, blobs[i]->h_, blobs[i]->w_, blobs[i]->c_, n_out);
			//load kernel and bias
			mr->get_conv_kb_cpu(i, tmp_cv);
			layers[i] = tmp_cv;
			blobs[i + 1] = new Blob(n_out, tmp_cv->out_h_, tmp_cv->out_w_);
		}
		else if (type == 1) {
			cout << "building id: " << i << " pool: " << endl;
			PoolLayer *tmp_pl = new PoolLayer(k_size, stride, blobs[i]->h_, blobs[i]->w_, blobs[i]->c_);
			layers[i] = tmp_pl;
			blobs[i + 1] = new Blob(blobs[i]->c_, tmp_pl->out_h_, tmp_pl->out_w_);

		}
		else if (type == 2) {
			cout << "building id: " << i << " relu: " << endl;
			ReluLayer *tmp_re = new ReluLayer(blobs[i]->c_, blobs[i]->h_, blobs[i]->w_);
			layers[i] = tmp_re;
			//in-place
			blobs[i + 1] = blobs[i];
		}
	}

	out_buff = new Blob(c, h, w);
	input_diff = new Blob(c, h, w);
}

void build_G_P_A_gpu() {
	//build G
	cout << "building G" << endl;
	for (int i = 0; i<n_blob; i++) {
		G[i] = new Blob(1, blobs[i]->c_, blobs[i]->c_, true, false);
	}

	//build P and A
	cout << "building P and A" << endl;
	//Blob *tpb = blobs[n_blob-1];
	Blob *tpb = blobs[content_id + 1];
	P = new Blob(tpb->c_, tpb->h_, tpb->w_, true, false);

	for (int i = 0; i<style_spots; i++) {
		Blob *tmp = G[style_after_id[i] + 1];
		A[style_after_id[i] + 1] = new Blob(1, tmp->h_, tmp->w_, true, false);
	}
}
void build_G_P_A_cpu() {
	//build G
	cout << "building G" << endl;
	for (int i = 0; i<n_blob; i++) {
		G[i] = new Blob(1, blobs[i]->c_, blobs[i]->c_);
	}

	//build P and A
	cout << "building P and A" << endl;
	//Blob *tpb = blobs[n_blob-1];
	Blob *tpb = blobs[content_id + 1];
	P = new Blob(tpb->c_, tpb->h_, tpb->w_);

	for (int i = 0; i<style_spots; i++) {
		Blob *tmp = G[style_after_id[i] + 1];
		A[style_after_id[i] + 1] = new Blob(1, tmp->h_, tmp->w_);
	}
}

void get_G_P_A_gpu(){
	//get P A
	blobs[0]->set_gpu_data(style);
	for (int i = 0; i<n_layer; i++) {
		layers[i]->forward_gpu(blobs[i], blobs[i + 1]);
		if (A[i + 1] != NULL) {
			get_style_G_gpu(blobs[i + 1], A[i + 1]);
			cout << "built A_" << i << endl;
		}
	}

	//clear the ghost!!
	for (int i = 0; i<n_blob; i++) {
		blobs[i]->clear_gpu();
	}
	blobs[0]->set_gpu_data(content);
	for (int i = 0; i<n_layer; i++) {
		layers[i]->forward_gpu(blobs[i], blobs[i + 1]);
	}
	cudaMemcpy(P->gpu_data, blobs[content_id + 1]->gpu_data, sizeof(float)*blobs[content_id + 1]->count_, cudaMemcpyDeviceToDevice);
	cout << "built P" << endl;

	//clear the ghost!!
	for (int i = 0; i<n_blob; i++) {
		blobs[i]->clear_gpu();
	}
}
void get_G_P_A_cpu(){
	//get P A
	memcpy(blobs[0]->cpu_data, style, sizeof(float)*blobs[0]->count_);
	for (int i = 0; i<n_layer; i++) {
		layers[i]->forward_cpu(blobs[i], blobs[i + 1]);
		if (A[i + 1] != NULL) {
			get_style_G_cpu(blobs[i + 1], A[i + 1]);
			cout << "built A_" << i << endl;
		}
	}

	//clear the ghost!!
	for (int i = 0; i<n_blob; i++) {
		memset(blobs[i]->cpu_data, 0, sizeof(float)*blobs[i]->count_);
		memset(blobs[i]->cpu_diff, 0, sizeof(float)*blobs[i]->count_);
	}

	memcpy(blobs[0]->cpu_data, content, sizeof(float)*blobs[0]->count_);
	for (int i = 0; i<n_layer; i++) {
		layers[i]->forward_cpu(blobs[i], blobs[i + 1]);
	}

	memcpy(P->cpu_data, blobs[content_id + 1]->cpu_data, sizeof(float)*blobs[content_id + 1]->count_);
	cout << "built P" << endl;


	//clear the ghost!!
	for (int i = 0; i<n_blob; i++) {
		memset(blobs[i]->cpu_data, 0, sizeof(float)*blobs[i]->count_);
		memset(blobs[i]->cpu_diff, 0, sizeof(float)*blobs[i]->count_);
	}
}

void do_iterate_gpu() {
	//copy content to input blob
	//memcpy(blobs[0]->cpu_data,content,sizeof(float)*blobs[0]->count_);
	blobs[0]->set_gpu_data(content);

	//out_buff = new Blob(c, h, w);
	//input_diff = new Blob(c, h, w, true, true);
	int count = 1;
	for (int i = 0; i<max_iterations; i++) {
		cout << "iter: " << i << endl;
		float loss = 0;
		for (int j = 0; j<n_layer; j++) {
			layers[j]->forward_gpu(blobs[j], blobs[j + 1]);
			if (j == content_id) {
				if (!produce) {
				}
			}
			if (A[j + 1] != NULL) {
				get_style_G_gpu(blobs[j + 1], G[j + 1]);
				if (!produce) {
				}
			}
		}
		//clear diff for back prop
		for (int k = 0; k<n_blob; k++) {
			//memset(blobs[k]->cpu_diff,0,sizeof(float)*(blobs[k]->count_));
			blobs[k]->clear_gpu(false, true);
		}
		//back prop
		for (int j = n_layer - 1; j >= 0; j--) {
			if (A[j + 1] != NULL) {
				add_style_grad_gpu(blobs[j + 1], G[j + 1], A[j + 1], style_factor*stws[j + 1] / style_spots);
			}
			if (j == content_id) {
				add_content_grad_gpu(blobs[content_id + 1], P, content_factor);
			}

			layers[j]->backward_gpu(blobs[j], blobs[j + 1]);
		}

		//adjust input
		axpy_gpu(blobs[0]->count_, momentum - 1, input_diff->gpu_data, input_diff->gpu_data);
		axpy_gpu(blobs[0]->count_, rate, blobs[0]->gpu_diff, input_diff->gpu_data);
		axpy_gpu(blobs[0]->count_, -1, input_diff->gpu_data, blobs[0]->gpu_data);

		if (!produce) {
		}
		count++;
		*kernel_progress = 100.0*count/max_iterations;
		//anneal learning rate
		if (count%count_fold == 0) {
			rate /= rate_fold;
		}
	}
	blobs[0]->get_gpu_data(out_buff->cpu_data);
}
void do_iterate_cpu() {
	//copy content to input blob
	memcpy(blobs[0]->cpu_data, content, sizeof(float)*blobs[0]->count_);
	//out_buff = new Blob(c, h, w);
	//input_diff = new Blob(c, h, w);
	int count = 0;
	for (int i = 0; i<max_iterations; i++) {
		cout << "iter: " << i << endl;
		float loss = 0;
		for (int j = 0; j<n_layer; j++) {
			layers[j]->forward_cpu(blobs[j], blobs[j + 1]);
			if (j == content_id) {
				if (!produce) {
					loss += content_factor*get_content_loss(blobs[content_id + 1], P);
				}
			}
			if (A[j + 1] != NULL) {
				get_style_G_cpu(blobs[j + 1], G[j + 1]);
				if (!produce) {
					loss += style_factor*get_style_loss(blobs[j + 1], G[j + 1], A[j + 1]);
				}
			}
		}
		//clear diff for back prop
		for (int k = 0; k<n_blob; k++) {
			memset(blobs[k]->cpu_diff, 0, sizeof(float)*(blobs[k]->count_));
		}
		//back prop
		for (int j = n_layer - 1; j >= 0; j--) {
			if (A[j + 1] != NULL) {
				add_style_grad_cpu(blobs[j + 1], G[j + 1], A[j + 1], style_factor*stws[j + 1] / style_spots);
			}
			if (j == content_id) {
				add_content_grad_cpu(blobs[content_id + 1], P, content_factor);
			}
			layers[j]->backward_cpu(blobs[j], blobs[j + 1]);
		}
		//adjust input
		for (int j = 0; j<blobs[0]->count_; j++) {
			input_diff->cpu_diff[j] = momentum*(input_diff->cpu_diff[j]) + rate*(blobs[0]->cpu_diff[j]);
			blobs[0]->cpu_data[j] -= input_diff->cpu_diff[j];
		}

		if (!produce) {
			cout << "loss: " << loss << endl;
		}
		count++;
		*kernel_progress = 105.0*count/max_iterations;
		//anneal learning rate
		if (count%count_fold == 0) {
			rate /= rate_fold;
		}
	}
	memcpy(out_buff->cpu_data, blobs[0]->cpu_data, sizeof(float)*(out_buff->count_));
}

void destroy_gpu() {

}
void destroy_cpu() {

}

//load model;
//check gpu;
//setup cublas if use gpu;
//alloc memory;
//build net;
int kernel_setup_env(bool force_cpu) {
	//CUDA_GET_DEVICE_COUNT
	int device_count = 0;
	int return_value = 0;
	cudaGetDeviceCount(&device_count);
	if (device_count>0) {
		use_gpu = true;
		return_value = USE_GPU;
	}
	else {
		use_gpu = false;
		return_value = USE_CPU;
	}
	if (force_cpu) {
		use_gpu = false;
		return_value = USE_CPU;
	}
	mr = new ModelReader(".\\model\\VGG19_cut_after_relu_3_1.dat");
	c = 3;
	if (use_gpu) {
		h = gpu_a;
		w = gpu_a;
		setup_gpu();
		//handle error: insufficient memory
		build_net_gpu();
		build_G_P_A_gpu();
	}
	else {
		h = cpu_a;
		w = cpu_a;
		setup_cpu();
		//handle error: insufficient memory
		build_net_cpu();
		build_G_P_A_cpu();
	}
	return return_value;
}

//all matrices should be allocated before call;
//make sure:
//c = 3;
//if use gpu, h=w=320;
//if use cpu, h=w=200;
int kernel_generate(float *content_src, float *style_src, int iter, float style_rate, float *progress, float *generate_dst) {
	rate = rate_origin;
	content = content_src;
	style = style_src;
	max_iterations = iter;
	style_factor = style_rate;

	kernel_progress = progress;
	//experiment
	content_factor = (1-style_factor)/50000;
	if (use_gpu) {
		get_G_P_A_gpu();
		do_iterate_gpu();
		memcpy(generate_dst, out_buff->cpu_data, sizeof(float)*out_buff->count_);
	}
	else {
		get_G_P_A_cpu();
		do_iterate_cpu();
		memcpy(generate_dst, out_buff->cpu_data, sizeof(float)*out_buff->count_);
	}
	return 0;
}

//detroy model;
//destroy cublas if use gpu;
//free memory;
int kernel_destroy_env() {

}
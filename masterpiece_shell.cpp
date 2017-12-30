#include <opencv2/opencv.hpp>
#include "masterpiece_shell.h"
#include "masterpiece_kernel.h"
#include <stdlib.h>
using namespace cv;

float mean_c[3] = { 104,117,123 };
int output_max_dim = 512;

bool use_gpu_shell;
int proc_a;

int out_h;
int out_w;

Mat content_origin;
Mat content_out_mix;
Mat content_proc_mat;
Mat style_origin;
Mat style_proc_mat;

bool content_loaded = false;
bool style_loaded = false;
bool env_set = false;

float *content_proc;//should free
float *style_proc;//should free
float *generate_proc;//should free

Mat generate_proc_mat;
Mat generate_out_mix;
Mat output;

void get_out_size(int in_h, int in_w) {
	if (in_h >= in_w) {
		out_h = output_max_dim;
		out_w = (int)(1.0*out_h*in_w / in_h);
	}
	else {
		out_w = output_max_dim;
		out_h = (int)(1.0*out_w*in_h / in_w);
	}
}

//make sure size compatibility!
//channel mean is substracted here.
void mat_to_float(Mat& mat, float *dst) {
	float *it = dst;
	unsigned char *data = mat.data;
	for (int c_ = 0; c_<mat.channels(); c_++) {
		for (int i = 0; i<mat.rows*mat.cols; i++) {
			*(it) = (float)(data[mat.channels()*i + c_] - mean_c[c_]);
			it++;
		}
	}
}

//make sure size compatibility!
//channel mean is added here.
void float_to_mat(float *src, Mat& mat) {
	unsigned char *data = mat.data;
	for (int c_ = 0; c_<mat.channels(); c_++) {
		for (int i = 0; i<mat.rows*mat.cols; i++) {
			float pix = (*src) + mean_c[c_];
			pix = pix<0 ? 0 : pix;
			pix = pix>255 ? 255 : pix;
			data[i*mat.channels() + c_] = (unsigned char)pix;
			src++;
		}
	}
}

int shell_init_env(bool force_cpu) {
	int ret_kernel_init = kernel_setup_env(force_cpu);
	if (ret_kernel_init == USE_GPU) {
		use_gpu_shell = true;
		proc_a = gpu_a;
	}
	else {
		use_gpu_shell = false;
		proc_a = cpu_a;
	}
	content_proc = (float*)malloc(sizeof(float)*channel*proc_a*proc_a);
	style_proc = (float*)malloc(sizeof(float)*channel*proc_a*proc_a);
	generate_proc = (float*)malloc(sizeof(float)*channel*proc_a*proc_a);
	generate_proc_mat.create(proc_a, proc_a, CV_8UC(channel));
	env_set = true;
	return 0;
}

int shell_load_content(char *path, unsigned char *content_show) {
	//TODO: return show image
	content_origin = imread(path);
	if (!content_origin.data) {
		return LOAD_FAILURE;
	}
	get_out_size(content_origin.rows, content_origin.cols);
	resize(content_origin, content_out_mix,Size(out_w,out_h), (0, 0), (0, 0), INTER_CUBIC);
	resize(content_origin, generate_out_mix, Size(out_w, out_h), (0, 0), (0, 0), INTER_CUBIC);
	resize(content_origin, content_proc_mat, Size(proc_a, proc_a), (0, 0), (0, 0), INTER_CUBIC);
	mat_to_float(content_proc_mat, content_proc);
	content_loaded = true;
	return 0;
}

int shell_load_style(char *path, unsigned char *style_show) {
	//TODO: return show image
	style_origin = imread(path);
	if (!style_origin.data) {
		return LOAD_FAILURE;
	}
	resize(style_origin, style_proc_mat, Size(proc_a, proc_a), (0, 0), (0, 0), INTER_CUBIC);
	mat_to_float(style_proc_mat, style_proc);
	style_loaded = true;
	return 0;
}

int shell_do_generate(int iter_num, float style_factor, float *progress) {
	if (!(env_set&&style_loaded&&content_loaded)) {
		return CANT_DO_GENERATE;
	}
	kernel_generate(content_proc, style_proc, iter_num, style_factor, progress, generate_proc);
	float_to_mat(generate_proc, generate_proc_mat);
	resize(generate_proc_mat, generate_out_mix, Size(out_w, out_h), (0, 0), (0, 0), INTER_CUBIC);
	addWeighted(content_out_mix, 0, generate_out_mix, 1, 0.0, output);
	return 0;
}

int shell_get_mix(float style_weight, unsigned char *mix_show) {
	addWeighted(content_out_mix, 1-style_weight, generate_out_mix, style_weight, 0.0, output);
	//temp
	imshow("mix", output);
	waitKey(0);
	//temp
	return 0;
}

int shell_save_output(char *path) {
	imwrite(path, output);
	return 0;
}

int shell_destroy_env() {
	return 0;
}
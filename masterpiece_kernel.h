#pragma once
#define USE_GPU 0
#define USE_CPU 1
#define CPU_MEM_OUT -1
#define GPU_MEM_OUT -2
#define GEN_SUCCESS 0
#define GEN_FAILURE -1

#define channel 3
#define cpu_a 240
#define gpu_a 360

//load model;
//check gpu;
//setup cublas if use gpu;
//alloc memory;
//build net;
int kernel_setup_env(bool force_cpu);

//all matrices should be allocated before call;
//make sure:
//c = 3;
//if use gpu, h=w=320;
//if use cpu, h=w=200;
int kernel_generate(float *content, float *style, int iter, float style_rate, float *progress, float *generate_dst);

//detroy model;
//destroy cublas if use gpu;
//free memory;
int kernel_destroy_env();
#pragma once
#define LOAD_FAILURE -1
#define CANT_DO_GENERATE -2

int shell_init_env(bool force_cpu);

int shell_load_content(char *path,unsigned char *content_show);

int shell_load_style(char *path,unsigned char *style_show);

int shell_do_generate(int iter_num, float style_factor, float *progress);

int shell_get_mix(float style_weight, unsigned char *mix_show);

int shell_save_output(char *path);

int shell_destroy_env();
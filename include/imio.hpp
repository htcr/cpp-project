#ifndef IMIO_HPP
#define IMIO_HPP

int img_in(char* file_path, float** dst, int* c, int* h, int* w);
int noise(float* dst, int c, int h, int w);
int img_out(char* file_path, float* src, int c, int h, int w);
int img_show(float* src, int c, int h, int w);
void sub_mean(float* src, float* mean, int c, int h, int w);
void add_mean(float* src, float mean, int c, int h, int w);
//void linear_map(float* src, int c, int h, int w);
#endif

#ifndef LOSS_HPP
#define LOSS_HPP

#include "utils.hpp"

void add_content_grad_cpu(Blob *F, Blob *P,float scale);

void get_style_G_cpu(Blob *F, Blob *G);

void add_style_grad_cpu(Blob *F, Blob *G, Blob *A, float scale);

//gpu version

void add_content_grad_gpu(Blob *F, Blob *P,float scale);/*untested!*/

void get_style_G_gpu(Blob *F, Blob *G);/*untested!*/

void add_style_grad_gpu(Blob *F, Blob *G, Blob *A, float scale);/*untested!*/



float get_content_loss(Blob *F, Blob *P);

float get_style_loss(Blob *F, Blob *G, Blob *A);

#endif
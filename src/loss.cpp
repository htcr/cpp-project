extern "C"{
	#include <cblas.h>
}

#include "loss.hpp"
#include "utils.hpp"


//clear diff before calling
void add_content_grad_cpu(Blob *F, Blob *P, float scale){
	//cblas_saxpy(F->count_, scale,F->cpu_data,1,F->cpu_diff,1);
	//cblas_saxpy(P->count_,-scale,P->cpu_data,1,F->cpu_diff,1);
	axpy_cpu(F->count_,scale,F->cpu_data,F->cpu_diff);
	axpy_cpu(P->count_,-scale,P->cpu_data,F->cpu_diff);
}

void add_content_grad_gpu(Blob *F, Blob *P, float scale){
	axpy_gpu(F->count_,scale,F->gpu_data,F->gpu_diff);
	axpy_gpu(P->count_,-scale,P->gpu_data,F->gpu_diff);
}

//compute G from F, store to G
void get_style_G_cpu(Blob *F, Blob *G){
	int k = (F->h_)*(F->w_);
	/*
	cblas_sgemm(CblasRowMajor,
				CblasNoTrans,CblasTrans,
				F->c_,F->c_,k,
				1.0,
				F->cpu_data,k,
				F->cpu_data,k,
				0.0,
				G->cpu_data,F->c_);
	*/
	gemm_cpu(false,true,
			 F->c_,F->c_,k,
			 1.0,
			 F->cpu_data,F->cpu_data,
			 0.0,
			 G->cpu_data);
}

void get_style_G_gpu(Blob *F, Blob *G){
	int k = (F->h_)*(F->w_);
	gemm_gpu(false,true,
			 F->c_,F->c_,k,
			 1.0,
			 F->gpu_data,F->gpu_data,
			 0.0,
			 G->gpu_data);
}

//compute gradient and add to F->cpu_diff,
//so upper gradients should be firstly propageted to F->cpu_diff.
//G will be ruined after this function
void add_style_grad_cpu(Blob *F, Blob *G, Blob *A, float scale){
	get_style_G_cpu(F,G);
	//cblas_saxpy(A->count_,-1,A->cpu_data,1,G->cpu_data,1);
	axpy_cpu(A->count_,-1,A->cpu_data,G->cpu_data);
	int n = (F->h_)*(F->w_);
	float s = (1.0*F->count_)*(F->count_)/scale;
	/*
	cblas_sgemm(CblasRowMajor,
				CblasTrans,CblasNoTrans,
				F->c_,n,F->c_,
				1.0/s,
				G->cpu_data,F->c_,
				F->cpu_data,n,
				1.0,
				F->cpu_diff,n);
	*/
	gemm_cpu(true,false,
			 F->c_,n,F->c_,
			 1.0/s,
			 G->cpu_data,F->cpu_data,
			 1.0,
			 F->cpu_diff);
}

void add_style_grad_gpu(Blob *F, Blob *G, Blob *A, float scale){
	get_style_G_gpu(F,G);
	axpy_gpu(A->count_,-1,A->gpu_data,G->gpu_data);
	int n = (F->h_)*(F->w_);
	float s = (1.0*F->count_)*(F->count_)/scale;
	gemm_gpu(true,false,
			 F->c_,n,F->c_,
			 1.0/s,
			 G->gpu_data,F->gpu_data,
			 1.0,
			 F->gpu_diff);
}

//we temporarily don't care about loss because it's not necessary for
//generating pics.
float get_content_loss(Blob *F, Blob *P){
	cblas_scopy(F->count_,F->cpu_data,1,P->cpu_diff,1);
	cblas_saxpy(F->count_,-1,P->cpu_data,1,P->cpu_diff,1);
	return 0.5*cblas_snrm2(F->count_,P->cpu_diff,1);
}

//call before add_style_grad
float get_style_loss(Blob *F, Blob *G, Blob *A){
	float scale = 0.25/(F->count_)/(F->count_);
	cblas_scopy(G->count_,G->cpu_data,1,G->cpu_diff,1);
	cblas_saxpy(G->count_,-1,A->cpu_data,1,G->cpu_diff,1);
	return scale*cblas_snrm2(G->count_,G->cpu_diff,1);
}
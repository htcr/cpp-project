#include <stdlib.h>
//#include <random>
#include <opencv2/opencv.hpp>
//#include <float.h>
#include "imio.hpp"

using namespace cv;
//using namespace std;

float mean_c[3] = {104,117,123};

int img_in(char* file_path, float** pdst, int* c, int* h, int* w){
	Mat img;
	img = imread(file_path);
	if(!img.data){
		return -1;
	}
	unsigned char *data = img.data;
	*c = img.channels();
	*h = img.rows;
	*w = img.cols;
	int count = (*c)*(*h)*(*w);
	*pdst = (float*) malloc(count*sizeof(float));
	float *it = *pdst;
	for(int c_ = 0;c_<img.channels();c_++){
		for(int i = 0;i<img.rows*img.cols;i++){
			*(it) = (float)(data[img.channels()*i+c_]-mean_c[c_]);
			it++;
		}
	}
	return 0;
}

/*
int noise(float* dst, int c, int h, int w){
	default_random_engine e;
	uniform_real_distribution<float> u(0,10);
	for(int i = 0;i<c*h*w;i++){
		dst[i] = u(e);
	}
	return 0;
}
*/


int img_out(char* file_path, float* src, int c, int h, int w){
	//linear_map(src,c,h,w);
	Mat img;
	img.create(h,w,CV_8UC(c));
	unsigned char *data = img.data;
	for(int c_ = 0;c_<c;c_++){
		for(int i = 0;i<h*w;i++){
			float pix = (*src)+mean_c[c_];
			pix = pix<0?0:pix;
			pix = pix>255?255:pix;
			data[i*c+c_] = (unsigned char)pix;
			src++;
		}
	}
	imwrite(file_path,img);
	return 0;
}

int img_show(float* src, int c, int h, int w){
	Mat img;
	img.create(h,w,CV_8UC(c));
	unsigned char *data = img.data;
	for(int c_ = 0;c_<c;c_++){
		for(int i = 0;i<h*w;i++){
			float pix = (*src)+mean_c[c_];
			pix = pix<0?0:pix;
			pix = pix>255?255:pix;
			data[i*c+c_] = (unsigned char)pix;
			src++;
		}
	}
	namedWindow("imshow", WINDOW_AUTOSIZE);
	imshow("imshow", img);
	waitKey(0);
	return 0;
}

void sub_mean(float* src, float* mean, int c, int h, int w){
	float sum = 0;
	for(int i = 0;i<c*h*w;i++){
		sum+=src[i];
	}
	sum/=(c*h*w);
	*mean = sum;
	for(int i = 0;i<c*h*w;i++){
		src[i]-=sum;
	}
}

void add_mean(float* src, float mean, int c, int h, int w){
	for(int i = 0;i<c*h*w;i++){
		src[i]+=mean;
	}
}

/*
void linear_map(float* src, int c, int h, int w){
	for(int c_ = 0; c_<c;c_++){
		float min = FLT_MAX;
		float max = FLT_MIN;
		for(int h_ = 0;h_<h;h_++){
			for(int w_ = 0;w_<w;w_++){
				float pix = src[h_*w+w_];
				min = (pix<min)?(pix):(min);
				max = (pix>max)?(pix):(max);
			}	
		}
		float d = max-min;
		for(int h_ = 0;h_<h;h_++){
			for(int w_ = 0;w_<w;w_++){
				float pix = src[h_*w+w_];
				src[h_*w+w_] = (255.0/d*(pix-min));
			}	
		}
		src+=(h*w);
	}
}
*/
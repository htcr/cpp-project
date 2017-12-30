#ifndef MODELREADER_HPP
#define MODELREADER_HPP

#include "layers.hpp"
//#include "caffe.pb.h"
#include <string>
using namespace std;



class ModelReader{
public:
	//caffe::NetParameter net_param_;
	float* net_param_;

public:
	ModelReader(string model_path);
	//void print_layer_id();
	int get_conv_param(int id,int *pad, int *k_size, int *stride, int *n_out);
	void get_conv_kb_cpu(int id,ConvLayer *cv);
	void get_conv_kb_gpu(int id,ConvLayer *cv);/*untested!*/

private:
	bool ReadProtoFromBinaryFile(const char* filename, float** buff);
};

#endif
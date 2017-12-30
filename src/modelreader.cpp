//#include "caffe.pb.h"
#include "modelreader.hpp"

//#include <fcntl.h>

//#include <boost/filesystem.hpp>
//#include <iomanip>

#include <iostream>
#include <fstream>
//#include <stdint.h>

//#include <google/protobuf/io/coded_stream.h>
//#include <google/protobuf/io/zero_copy_stream_impl.h>
//#include <google/protobuf/text_format.h>

/*
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

const int kProtoReadBytesLimit = INT_MAX;
*/

ModelReader::ModelReader(string model_path){

	/*
	cout<<"ModelReader: building fstream from: "<<model_path<<endl;	
	this->input_ = new fstream(model_path.data(),ios::in|ios::binary);
	cout<<"ModelReader: building fstream finished;"<<endl;
	*/

	bool success = ReadProtoFromBinaryFile(model_path.data(),&net_param_);

	//if(!(this->net_param_.ParseFromIstream(input_))){
	if(!success){	
		cout<<"ModelReader: failed to parse from: "<<model_path<<endl;
	}else{
		cout<<"ModelReader: successfully parsed from: "<<model_path<<endl;
		//cout<<"ModelReader: current model name: "<<net_param_.name()<<endl;	
	}
	
}

int  type[12] = {0,2,0,2,1,0,2,0,2,1,0,2};
int  k_count[12] = {1728,0,36864,0,0,73728,0,147456,0,0,294912,0};
int  b_count[12] = {64,0,64,0,0,128,0,128,0,0,256,0};
int  total = 555328;
int  param_count = 0;

bool ModelReader::ReadProtoFromBinaryFile(const char* filename, float** buff) {
	bool success;
	(*buff) = (float*)malloc(sizeof(float)*total);
	float* p = *buff;
	ifstream fin(filename, std::ios::binary);
	fin.read((char*)p, sizeof(float)*total);
	if (fin) {
		fin.close();
		return true;
	}
	else {
		return false;
	}
}

/*
void ModelReader::print_layer_id(){
	for(int i = 0;i<net_param_.layers_size();i++){
		caffe::V1LayerParameter tmp = net_param_.layers(i);
		cout<<tmp.name()<<" id: "<<i<<endl;
	}
}
*/

int ModelReader::get_conv_param(int id,int *pad, int *k_size, int *stride, int *n_out){
	if (type[id] == 0) {
		*pad = 1;
		*k_size = 3;
		*stride = 1;
		*n_out = b_count[id];
		return 0;
	}
	else if (type[id] == 1) {
		*pad = 0;
		*k_size = 2;
		*stride = 2;
		return 1;
	}
	else if (type[id] == 2) {
		return 2;
	}
	return -1;
	/*
	caffe::V1LayerParameter tmp = net_param_.layers(id);
	string name = tmp.name();
	char type = name[0];
	if(type=='c'){
		caffe::ConvolutionParameter cv = tmp.convolution_param();
		*pad = 1;
		*k_size = 3;
		*stride = 1;
		*n_out = cv.num_output();
		return 0;
	}else if(type=='p'){
		*pad = 0;
		*k_size = 2;
		*stride = 2;
		return 1;
	}else if(type=='r'){
		return 2;
	}
	return -1;
	*/
}

void ModelReader::get_conv_kb_cpu(int id,ConvLayer *cv){
	for (int i = 0; i<k_count[id]; i++) {
		cv->kernel_->cpu_data[i] = net_param_[param_count++];
	}
	cout << "loaded " << k_count[id] << " floats for k of layer " << id << endl;
	for (int i = 0; i<b_count[id]; i++) {
		cv->bias_->cpu_data[i] = net_param_[param_count++];
	}
	cout << "loaded " << b_count[id] << " floats for b of layer " << id << endl;
	/*
	caffe::V1LayerParameter tmp = net_param_.layers(id);
	caffe::BlobProto k = tmp.blobs(0);
	caffe::BlobProto b = tmp.blobs(1);
	for(int i = 0;i<k.data_size();i++){
		cv->kernel_->cpu_data[i] = k.data(i);
	}
	cout<<"loaded "<<k.data_size()<<" floats for k of "<<tmp.name()<<endl;
	for(int i = 0;i<b.data_size();i++){
		cv->bias_->cpu_data[i] = b.data(i);
	}
	cout<<"loaded "<<b.data_size()<<" floats for b of "<<tmp.name()<<endl;
	*/
}

void ModelReader::get_conv_kb_gpu(int id,ConvLayer *cv){
	cv->kernel_->alloc_cpu_data();
	cv->bias_->alloc_cpu_data();
	for (int i = 0; i<k_count[id]; i++) {
		cv->kernel_->cpu_data[i] = net_param_[param_count++];
	}
	cv->kernel_->set_gpu_data(cv->kernel_->cpu_data);
	cout << "loaded " << k_count[id] << " floats for k of layer " << id << endl;
	for (int i = 0; i<b_count[id]; i++) {
		cv->bias_->cpu_data[i] = net_param_[param_count++];
	}
	cv->bias_->set_gpu_data(cv->bias_->cpu_data);
	cout << "loaded " << b_count[id] << " floats for b of layer " << id << endl;
	cv->kernel_->free_cpu_data();
	cv->bias_->free_cpu_data();
}
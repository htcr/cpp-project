#include "pipe.h"
#include "masterpiece_shell.h"
#include <iostream>
#include <fstream>
#include <Windows.h>
using namespace std;

char pipe_folder_path[500] = {0};
char pipe_ui2kernel_name[30] = "UI2K";
char pipe_kernel2ui_name[30] = "K2UI";
char pipe_show_img_name[30] = "show.jpg";

char pipe_path_buff[500] = {0};

int prev_rmsg_id = 0;
int cur_rmsg_id = 0;
int wmsg_id = 0;
char pipe_msg_buff[1000] = {0};
float pipe_progress = 0;
int show_id = 0;

bool generating = false;

void clear(char *str,int size){
	memset(str,0,size);
}

void pipe_kernel2ui_create(){
	cout<<"请用一行输入UWP界面中显示的文件路径，用于kernel和ui的数据交换；不要添加额外的字符。"<<endl;
	fgets(pipe_folder_path,500,stdin);
	(*strchr(pipe_folder_path,'\n')) = 0;
	//cout<<pipe_folder_path<<endl;
	sprintf(pipe_path_buff,"%s\\%s",pipe_folder_path,pipe_kernel2ui_name);
	ofstream out(pipe_path_buff, ios::out);
	if (out){
		out.close();
		cout<<"pipe_k2ui created."<<endl;
	}
	
	/*
	fstream pipe_k2ui;
     pipe_k2ui.open(pipe_path_buff,ios::in);
     if(!pipe_k2ui){
         //k2ui not found, create one
		 cout<<pipe_path_buff<<"没有被创建";
     }
      else{
		 //exist, 
         cout<<pipe_path_buff<<"已经存在";
     }
	 */
	clear(pipe_path_buff,500);
}

bool pipe_ui2kernel_exist(){
	sprintf(pipe_path_buff,"%s\\%s",pipe_folder_path,pipe_ui2kernel_name);
	ofstream out(pipe_path_buff, ios::out);
	if (out){
		out.close();
		//cout<<"pipe_k2ui created."<<endl;
	}
	/*
	fstream ui2k;
	ui2k.open(pipe_path_buff,ios::in);
	if(!ui2k){
		//cout<<"waiting for ui connection."<<endl;
		clear(pipe_path_buff,500);
		return false;
	}else{
		ui2k.close();
		clear(pipe_path_buff,500);
		return true;
	}*/
	clear(pipe_path_buff,500);
	return true;
}

bool pipe_new_msg(){
	sprintf(pipe_path_buff,"%s\\%s",pipe_folder_path,pipe_ui2kernel_name);
	ifstream in(pipe_path_buff);
	if (in)
	{		
		//in >> pipe_msg_buff;
		in.getline(pipe_msg_buff,1000);
		in.close();
	}
	cur_rmsg_id = atoi(pipe_msg_buff);
	if(cur_rmsg_id==prev_rmsg_id){
		clear(pipe_path_buff,500);
		clear(pipe_msg_buff,1000);
		return false;
	}else{
		prev_rmsg_id = cur_rmsg_id;
		clear(pipe_path_buff,500);
		return true;
	}	
	//clear(pipe_msg_buff,1000);
}

char get_one_char(char **ext_ptr){
	char *ptr = *ext_ptr;
	char ret = *ptr;
	while((*ptr)!='|'&&(*ptr)!=0){
		ptr++;
	}
	if(*ptr=='|'){
		ptr++;
	}
	*ext_ptr = ptr;
	return ret;
}

void get_str(char **ext_ptr, char *str_buff){
	char *ptr = *ext_ptr;
	//now ptr at str begin
	int str_len = 0;
	while((*ptr)!='|'&&(*ptr)!=0){
		str_buff[str_len++] = (*ptr);
		ptr++;
	}
	if(*ptr=='|'){
		ptr++;
	}
	*ext_ptr = ptr;
}

int get_integer(char **ext_ptr){
	char *ptr = *ext_ptr;
	//now ptr at integer begin
	int num = atoi(ptr);
	while((*ptr)!='|'&&(*ptr)!=0){
		ptr++;
	}
	if(*ptr=='|'){
		ptr++;
	}
	*ext_ptr = ptr;
	return num;
}

float get_float(char **ext_ptr){
	char *ptr = *ext_ptr;
	float rate = atof(ptr);
	while((*ptr)!='|'&&(*ptr)!=0){
		ptr++;
	}
	if(*ptr=='|'){
		ptr++;
	}
	*ext_ptr = ptr;
	return rate;
}

char pipe_content_path[500] = {0};
char pipe_style_path[500] = {0};
int pipe_iter_num;
float pipe_style_factor;

char pipe_save_path[500] = {0};

DWORD WINAPI thread_iterate(LPVOID lpParam)
{
	shell_load_content(pipe_content_path,NULL);
	shell_load_style(pipe_style_path, NULL);
	shell_do_generate(pipe_iter_num,pipe_style_factor,&pipe_progress);
	//shell_get_mix(1, NULL);
	generating = false;
	char temp_path_buff[500] ={0};
	sprintf(temp_path_buff,"%s\\%s",pipe_folder_path,"show.jpg");
	shell_save_output(temp_path_buff);
	pipe_send_msg('f');
	return 0;
}

void pipe_exec_msg(){
	char *ptr = pipe_msg_buff;
	get_integer(&ptr);
	char msg_type = get_one_char(&ptr);

	if(msg_type=='g'){
		memset(pipe_content_path,0,500);
		memset(pipe_style_path,0,500);
		get_str(&ptr,pipe_content_path);
		get_str(&ptr,pipe_style_path);
		pipe_iter_num = get_integer(&ptr);
		pipe_style_factor = get_float(&ptr);
		cout<<"cp: "<<pipe_content_path<<endl;
		cout<<"sp: "<<pipe_style_path<<endl;
		cout<<"it: "<<pipe_iter_num<<endl;
		cout<<"sf: "<<pipe_style_factor<<endl;
		//system("pause");
		generating = true;
		LPDWORD dwThreadID = 0;
		HANDLE hThread = CreateThread(NULL, 0, thread_iterate, NULL, 0, dwThreadID);
		CloseHandle(hThread);
	}else if(msg_type=='s'){
		memset(pipe_save_path,0,500);
		get_str(&ptr,pipe_save_path);
		cout<<"sv: "<<pipe_save_path<<endl;
		shell_save_output(pipe_save_path);
	}else{
		
	}
	
	clear(pipe_msg_buff,1000);
}

void pipe_send_msg(int type){
	sprintf(pipe_path_buff,"%s\\%s",pipe_folder_path,pipe_kernel2ui_name);
	char out_msg_buff[500] = {0};
	wmsg_id++;
	if(type=='g'){
		ofstream out(pipe_path_buff, ios::out);
		if (out){
			sprintf(out_msg_buff,"%d|g|%f",wmsg_id,pipe_progress);
			out<<out_msg_buff;
			out.close();
		}
	}else if(type=='f'){
		ofstream out(pipe_path_buff, ios::out);
		if (out){
			sprintf(out_msg_buff,"%d|f|show.jpg",wmsg_id,pipe_folder_path);
			out<<out_msg_buff;
			out.close();
		}
	}else{
	}
	clear(pipe_path_buff,500);
}

void pipe_no_new(){
	if(generating){
		pipe_send_msg('g');
	}
}
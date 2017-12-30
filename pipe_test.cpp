#include "pipe.h"
#include "masterpiece_shell.h"
#include <iostream>
#include <Windows.h>
using namespace std;
int main(){
	shell_init_env(false);
	pipe_kernel2ui_create();
	while(true){
		if(pipe_ui2kernel_exist()){
			break;
		}else{
			cout<<"waiting for ui connection."<<endl;
			Sleep(1000);
		}
	}
	cout<<"ui connected."<<endl;
	while(true){
		if(pipe_new_msg()){
			cout<<"received new msg."<<endl;
			pipe_exec_msg();
		}else{
			cout<<"nothing new."<<endl;
			pipe_no_new();
		}
		Sleep(1000);
	}
	system("pause");
	return 0;
}
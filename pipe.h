#ifndef PIPE_HPP
#define PIPE_HPP

void pipe_kernel2ui_create();
bool pipe_ui2kernel_exist();
bool pipe_new_msg();//if recving new msg
void pipe_send_msg(int type);//send
void pipe_exec_msg();//execute message command
void pipe_no_new();

#endif
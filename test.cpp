#include "masterpiece_shell.h"
#include <iostream>
using namespace std;

int main() {
	//img_in("F:\\ubuntu exchange\\content1.jpg", &content, &c, &h, &w);
	//img_in("F:\\ubuntu exchange\\style1.jpg", &style, &c, &h, &w);
	shell_init_env(false);
	shell_load_content("F:\\ubuntu exchange\\test_content.jpg",NULL);
	shell_load_style("F:\\ubuntu exchange\\task\\stock.jpg", NULL);
	float progress;
	shell_do_generate(200, 0.1, &progress);
	shell_get_mix(1, NULL);
	//shell_get_mix(0.5, NULL);
	shell_save_output("F:\\ubuntu exchange\\task\\out16.jpg");
	return 0;
}
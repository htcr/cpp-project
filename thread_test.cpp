#include <stdlib.h>
#include <iostream>
#include <Windows.h>
using namespace std;

void call_from_thread() {
	
}

DWORD WINAPI ThreadFunction(LPVOID lpParam)
{
	while(true){
		Sleep(1000);
		cout << "this is not main thread." << endl;
	}
	return 0;
}
int main() {
	LPDWORD dwThreadID = 0;
	HANDLE hThread = CreateThread(NULL, 0, ThreadFunction, NULL, 0, dwThreadID);
	CloseHandle(hThread);
	cout << "this is main thread." << endl;
	system("pause");
	return 0;
}
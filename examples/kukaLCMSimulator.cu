/***
nvcc -std=c++11 -o kukaSim.exe kukaLCMSimulator.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -llcm -gencode arch=compute_61,code=sm_61 -rdc=true -O3
***/
#define USE_WAFR_URDF 1
#define MPC_MODE 1
#define USE_LCM 1
#define PLANT 4
#include "../config.cuh"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

__host__ __forceinline__
bool tryParse(std::string& input, int& output) {
	try{output = std::stoi(input);}
	catch (std::invalid_argument) {return false;}
	return true;
}
__host__ __forceinline__
int getInt(int maxInt, int minInt){
	std::string input;	std::string exitCode ("q"); int x;
	while(1){
		getline(std::cin, input);
		while (!tryParse(input, x)){
			if (input.compare(input.size()-1,1,exitCode) == 0){return -1;}
				std::cout << "Bad entry. Enter a NUMBER\n";	getline(std::cin, input);
			}
		if (x >= minInt && x <= maxInt){break;}
		else{std::cout << "Entry must be in range[" << minInt << "," << maxInt << "]\n";}
	}
	return x;
}
__host__ __forceinline__
void keyboardHold(){
   	printf("Press enter to launch the simulator\n");	std::string input;	getline(std::cin, input);
}

template <typename T>
__host__ __forceinline__
void loadX(T *xk, int mode = 0){
	#define PI 3.14159
	if (mode == 1){
		xk[0] = PI/2.0; 	xk[1] = -PI/6.0; 	xk[2] = -PI/3.0; 	xk[3] = -PI/2.0; 	xk[4] = 3.0*PI/4.0; 	xk[5] = -PI/4.0; 	xk[6] = 0.0;
	}
	else{
		for(int i = 0; i < NUM_POS; i++){xk[i] = 0.0;}	
	}
	for(int i = NUM_POS; i < STATE_SIZE; i++){xk[i] = 0.0;}
}

int main(int argc, char *argv[])
{
	// Ask for user input on the initial state
	printf("What initial state would you like? (0: 0s, 1: Patrick Pos)\n");
	double xInit[STATE_SIZE];	loadX(xInit,getInt(1,0));
	// then get user input on simulator parameters
	printf("What should the simulator rate be (in hz)?\n");
	int hz = getInt(10000,100);
	printf("How many substeps should the simulator take on each tick?\n");
	int steps = getInt(1000,1);
	// then ask for debug mode
	printf("What debug mode would you like? (0: None, 1:dt,uk,xk,xkp1 2:eePos, 3:qk|qdk)\n");
	int debug = getInt(3,0);
	// then launch simulator on enter
	keyboardHold();
	runLCMSimulator(xInit,steps,hz,debug); // defined in LCMHelpers
	return 1;
}
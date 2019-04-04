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

int main(int argc, char *argv[])
{
	// Ask for user input on the initial state
	double xInit[STATE_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	// then get user input on simulator parameters
	printf("What should the simulator rate be (in hz)?\n");
	int hz = getInt(10000,100);
	printf("How many substeps should the simulator take on each tick?\n");
	int steps = getInt(1000,1);
	// then ask for debug mode
	printf("What debug mode would you like? (0: None, 1:dt,uk,xk,xkp1 2:eePos)\n");
	int debug = getInt(2,0);
	// then launch simulator on enter
	keyboardHold();
	runLCMSimulator(xInit); // defined in LCMHelpers
	return 1;
}
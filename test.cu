/***
nvcc -std=c++11 -o test.exe test.cu utils/cudaUtils.cu utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -rdc=true -O3
***/
#define EE_COST 1
#define PLANT 4
#include "DDPHelpers.cuh"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

int main(int argc, char *argv[]){
	double d_I[36*NUM_POS];
	double d_Tbody[36*NUM_POS];
	// double s_qddk[NUM_POS];
	double s_xk[STATE_SIZE];
	double s_uk[CONTROL_SIZE];
	initI<double>(d_I);
	initT<double>(d_Tbody);

	double s_I[36*NUM_POS]; // standard inertias -->  world inertias
	double s_Icrbs[36*NUM_POS]; // Icrbs inertias
	double s_J[6*NUM_POS]; // kinsol.J transformation matricies
	double s_temp[36*NUM_POS]; // temp work space (load Tbody mats into here) --> and JdotV
	double s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> and crm(twist) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
	double s_TA[36*NUM_POS]; // adjoint transpose --> crf(twist)
	double s_W[6*NUM_POS]; // to store net wrenches  
	double s_F[6*NUM_POS]; // to store forces in joint axis
	double s_JdotV[6*NUM_POS]; // JdotV vectors

	double deltas[] = {-0.66667,-0.33333,0.0,0.5,1.0};

	for (int d = 0; d < 5; d++){
		for (int i = 0; i < STATE_SIZE; i++){
			for (int j = 0; j < STATE_SIZE; j++){
				if(j == i){s_xk[j] = deltas[d];}
				else{s_xk[j] = 0;}
				printf("%f ",s_xk[j]);
			}
			printf("\n\n");
			for (int j = 0; j < CONTROL_SIZE; j++){s_uk[j] = 0;}
			load_Tb(s_xk,s_temp,d_Tbody,s_F,s_W);
			load_I(s_I,d_I);
			compute_T_TA_J(s_temp,s_temp2,s_TA,s_J);
			compute_Iw_Icrbs_twist(s_I,s_Icrbs,s_W,s_TA,s_J,s_xk,s_temp);
			compute_JdotV(s_JdotV,s_W,s_J,s_xk,s_temp);
			double *s_M = s_temp2;
			double *s_Tau = &s_temp2[2*NUM_POS*NUM_POS];
			compute_M_Tau(s_M, s_Tau, s_W, s_JdotV, s_F, s_Icrbs, s_W, s_J, s_I, s_xk, s_uk, s_temp, s_temp2, s_TA);
			printMat<double,NUM_POS,NUM_POS>(s_M,NUM_POS); printf("\n");
			printMat<double,1,NUM_POS>(s_Tau,1); printf("\n");
		}
	}
}
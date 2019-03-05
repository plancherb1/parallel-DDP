/*******
nvcc -std=c++11 -o testCostGrad.exe testCostGrad.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -gencode arch=compute_52,code=sm_52 -rdc=true -O3
*******/

#define PLANT 4
#define EE_COST 1
#include "../DDPHelpers.cuh"
#include <random>
#define NUM_REPS 1
#define ERR_TOL 0.01
#define RANDOM_MEAN 0
#define RANDOM_STDEVq 2
#define RANDOM_STDEVqd 5
std::default_random_engine randEng(time(0)); //seed
std::normal_distribution<double> randDistq(RANDOM_MEAN, RANDOM_STDEVq); //mean followed by stdiv
std::normal_distribution<double> randDistqd(RANDOM_MEAN, RANDOM_STDEVqd); //mean followed by stdiv

template <typename T>
__host__
void finiteDiffPos(T *x, T *eePosGrad, int ld_grad){
	T s_x[2*STATE_SIZE];	T eePos[2*6];
	#pragma unroll
	for (int diff_ind = 0; diff_ind < STATE_SIZE; diff_ind++){
		T *eeGrad = &eePosGrad[ld_grad*diff_ind];
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
		}
		// compute eePos and Vel
		compute_eePos_scratch<T>( s_x, 			    eePos);
		compute_eePos_scratch<T>(&s_x[STATE_SIZE], &eePos[6]);
		// now do finite diff rule
		#pragma unroll
		for (int i = 0; i < 6; i++){
			T deltaPos = eePos[i] - eePos[i+6];
			eeGrad[i]   = deltaPos / (2.0*FINITE_DIFF_EPSILON);
		}
	}
}

template <typename T>
__host__
void analyticalPos(T *x, T *eePosGrad){
	T s_cosq[NUM_POS];         	T s_sinq[NUM_POS];      	T s_eePos[6];
	T s_Tb[36*NUM_POS];			T d_Tb[36*NUM_POS]; 	   	initT<T>(d_Tb); 
	T s_T[36*NUM_POS];			T s_dT[36*NUM_POS];			T s_dTb[36*NUM_POS];
	compute_eePos<T>(s_T,s_eePos,s_dT,eePosGrad,s_sinq,s_Tb,s_dTb,x,s_cosq,d_Tb);
}

template <typename T>
__host__
void finiteDiffVel(T *x, T *eePosVelGrad, int ld_grad){
	T s_x[2*STATE_SIZE];	T eePos[2*6];	T eeVel[2*6];
	#pragma unroll
	for (int diff_ind = 0; diff_ind < STATE_SIZE; diff_ind++){
		T *eeGrad = &eePosVelGrad[ld_grad*diff_ind];
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
		}
		// compute eePos and Vel
		compute_eeVel_scratch<T>(s_x, 			   eePos, 	  eeVel);
		compute_eeVel_scratch<T>(&s_x[STATE_SIZE], &eePos[6], &eeVel[6]);
		// now do finite diff rule
		#pragma unroll
		for (int i = 0; i < 6; i++){
			T deltaPos = eePos[i] - eePos[i+6];
			T deltaVel = eeVel[i] - eeVel[i+6];
			eeGrad[i]   = deltaPos / (2.0*FINITE_DIFF_EPSILON);
			eeGrad[i+6] = deltaVel / (2.0*FINITE_DIFF_EPSILON);
		}
	}
}

template <typename T>
__host__
void analyticalVel(T *x, T *eePosVelGrad, int ld_grad){
	T s_cosq[NUM_POS];         	T s_sinq[NUM_POS];      	T s_Tb[36*NUM_POS];
	T d_Tb[36*NUM_POS]; 	   	initT<T>(d_Tb); 
	T s_Tb_dx[32*NUM_POS];	   	T s_TbTdt[32*NUM_POS];		T s_T[36*NUM_POS];
	T s_temp1[32*NUM_POS];	   	T s_temp2[32*NUM_POS];		T s_temp3[32*NUM_POS];
	T s_eePos[6];				T s_eeVel[6];
 	compute_eePosVel_dx<T>(x, s_Tb, d_Tb, s_cosq, s_sinq, s_Tb_dx, s_TbTdt, s_T, 
                           s_temp1, s_temp2, s_temp3, s_eePos, s_eeVel, eePosVelGrad, ld_grad);
}


template <typename T>
__host__
void loadAndClearPos(T *x, T *grad, T *grad2, int ld_grad){
	// load random state
	#pragma unroll
	for (int i=0; i < NUM_POS; i++){
		x[i] = static_cast<T>(randDistq(randEng));
		x[i+NUM_POS] = static_cast<T>(randDistqd(randEng));
	}
	// clear grads
	for (int i=0; i < ld_grad*STATE_SIZE; i++){
		grad[i] = 0;	grad2[i] = 0;
	}
}

template <typename T>
__host__
void testPos(){
	// allocate
	int ld_grad = 6;
	T *x =     (T *)malloc(        STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,ld_grad);
		// compute
		analyticalPos<T>(x,grad);
		finiteDiffPos<T>(x,grad2,ld_grad);
		// compare
		#pragma unroll
		for (int c = 0; c < STATE_SIZE; c++){
			#pragma unroll
			for (int r = 0; r < 6; r++){
				int ind = c*ld_grad + r;
				T err = abs(grad[ind] - grad2[ind]);
				if (err > ERR_TOL){
					printf("rep[%d] ind[%d]=c,r[%d,%d] has err[%.8f] for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,c,r,err,grad[ind],grad2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2); free(Tbody);
}

template <typename T>
__host__
void testVel(){
	// allocate
	int ld_grad = 12;
	T *x =     (T *)malloc(        STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,ld_grad);
		// compute
		analyticalVel<T>(x,grad,ld_grad);
		finiteDiffVel<T>(x,grad2,ld_grad);
		// compare
		#pragma unroll
		for (int c = 0; c < STATE_SIZE; c++){
			#pragma unroll
			for (int r = 0; r < 12; r++){
				int ind = c*ld_grad + r;
				T err = abs(grad[ind] - grad2[ind]);
				if (err > ERR_TOL){
					printf("rep[%d] ind[%d]=c,r[%d,%d] has err[%.8f] for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,c,r,err,grad[ind],grad2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2); free(Tbody);
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	char mode = '?';
	if (argc > 1){mode = argv[1][0];}
	switch (mode){
		case 'P':
			testPos<algType>();
			break;
		case 'V':
			testVel<algType>();
			break;
		default:
			printf("Input is [P]os [V]el or full [C]ost\n");
			return 1;
	}
	return 0;
}

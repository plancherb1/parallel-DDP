/*******
nvcc -std=c++11 -o testCostGrad.exe testCostGrad.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -gencode arch=compute_52,code=sm_52 -rdc=true -O3
*******/

#define PLANT 4
#define EE_COST 1
#define MPC_MODE 1
#define USE_WAFR_URDF 0
#define USE_EE_VEL_COST 1
#include "../DDPHelpers.cuh"
#include <random>
#define NUM_REPS 1
#define TEST_FINITE_DIFF_EPSILON 0.001
#define ERR_TOL 5 // in percent
#define RANDOM_MEAN 0
#define RANDOM_STDEVq 2
#define RANDOM_STDEVqd 5
#define RANDOM_STDEVu 50
#define RANDOM_STDEVg 2
std::default_random_engine randEng(time(0)); //seed
std::normal_distribution<double> randDistq(RANDOM_MEAN, RANDOM_STDEVq); //mean followed by stdiv
std::normal_distribution<double> randDistqd(RANDOM_MEAN, RANDOM_STDEVqd); //mean followed by stdiv
std::normal_distribution<double> randDistu(RANDOM_MEAN, RANDOM_STDEVu); //mean followed by stdiv
std::normal_distribution<double> randDistg(RANDOM_MEAN, RANDOM_STDEVg); //mean followed by stdiv

template <typename T>
__host__
void finiteDiffT(T *x, T *grad, int ld_grad){
	T s_x[2*STATE_SIZE];	T s_cosq[NUM_POS];		T s_sinq[NUM_POS];
	T s_Tb[36*NUM_POS];		T s_T[36*NUM_POS];		T s_T2[36*NUM_POS];
	T d_Tb[36*NUM_POS];		initT<T>(d_Tb);
	#pragma unroll
	for (int diff_ind = 0; diff_ind < NUM_POS; diff_ind++){
		T *gradc = &grad[ld_grad*diff_ind];
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? TEST_FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
		}
		// compute T
		load_Tb<T>(s_x,			    s_Tb,d_Tb,s_cosq,s_sinq);		compute_T_TA_J<T>(s_Tb,s_T);
		load_Tb<T>(&s_x[STATE_SIZE],s_Tb,d_Tb,s_cosq,s_sinq);		compute_T_TA_J<T>(s_Tb,s_T2);
		T *Tee = &s_T[36*(NUM_POS-1)];	T *Tee2 = &s_T2[36*(NUM_POS-1)];
		// now do finite diff rule
		#pragma unroll
		for (int i = 0; i < 16; i++){
			T delta  = Tee[i] - Tee2[i];
			gradc[i] = delta / (2.0*TEST_FINITE_DIFF_EPSILON);
		}
	}
}

template <typename T>
__host__
void analyticalT(T *x, T *grad){
	T s_cosq[NUM_POS];         	T s_sinq[NUM_POS];
	T s_Tb[36*NUM_POS];			T d_Tb[36*NUM_POS]; 	   	initT<T>(d_Tb); 
	T s_T[36*NUM_POS];			T s_dT[36*NUM_POS];			T s_dTb[36*NUM_POS];
   	load_Tb<T>(x,s_Tb,d_Tb,s_cosq,s_sinq,s_dTb);			
   	compute_T_TA_J<T>(s_Tb,s_T);								T *s_dTp = &s_dTb[16*NUM_POS];
   	compute_dT_dTA_dJ<T>(s_Tb,s_dTb,s_T,s_dT,s_dTp);
   	for (int k = 0; k < NUM_POS; k++){
		for (int i = 0; i < 16; i++){grad[16*k + i] = s_dT[36*k + i];}
	}
}

template <typename T>
__host__
void analyticalT2(T *x, T *grad){
	T s_cosq[NUM_POS];         	T s_sinq[NUM_POS];      		T s_temp1[32*NUM_POS];
	T s_Tb[36*NUM_POS];			T d_Tb[36*NUM_POS]; 	   		initT<T>(d_Tb); 
	T s_Tb_dx[16*NUM_POS];	   	T s_TbTdt[32*NUM_POS];			T s_T[36*NUM_POS];
	T s_Tb_dt_dx[32*NUM_POS];	T s_T_dt_dx[32*NUM_POS];		T s_T_dt_dx_prev[32*NUM_POS];
   	// load in Tb, Tb_dx, Tb_dt
    load_Tbdtdx<T>(x,s_Tb,d_Tb,s_sinq,s_cosq,s_Tb_dx,s_TbTdt,s_Tb_dt_dx);
    compute_T_TA_J<T>(s_Tb,s_T,nullptr,nullptr,s_TbTdt);
    T *s_Tb_dt = s_TbTdt;   T *s_T_dt = &s_TbTdt[16*NUM_POS];
    T *s_T_dx = s_temp1; 	T *s_T_dx_prev = &s_T_dx[16*NUM_POS];
    compute_T_dtdx<T>(s_Tb,s_Tb_dx,s_Tb_dt,s_Tb_dt_dx,s_T,s_T_dx,s_T_dt,s_T_dt_dx,s_T_dx_prev,s_T_dt_dx_prev);
   	for (int k = 0; k < NUM_POS; k++){
		for (int i = 0; i < 16; i++){grad[16*k + i] = s_T_dx[16*k + i];}
	}
}

template <typename T>
__host__
void finiteDiffTbdt(T *x, T *grad, int ld_grad){
	T s_x[2*STATE_SIZE];	T s_cosq[NUM_POS];		T s_sinq[NUM_POS];
	T s_Tb[36*NUM_POS];		T s_Tbdt[36*NUM_POS];	T s_Tbdt2[36*NUM_POS];
	T d_Tb[36*NUM_POS];		initT<T>(d_Tb);
	#pragma unroll
	for (int diff_ind = 0; diff_ind < STATE_SIZE; diff_ind++){
		T *gradc = &grad[ld_grad*diff_ind];
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? TEST_FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
		}
		// compute Tdt
		load_Tbdt<T>(s_x,			  s_Tb,d_Tb,s_cosq,s_sinq,s_Tbdt);
		load_Tbdt<T>(&s_x[STATE_SIZE],s_Tb,d_Tb,s_cosq,s_sinq,s_Tbdt2);
		int Tb_ind = diff_ind % NUM_POS;
		T *Tbeedt = &s_Tbdt[16*Tb_ind];	T *Tbeedt2 = &s_Tbdt2[16*Tb_ind];
		// now do finite diff rule
		#pragma unroll
		for (int i = 0; i < 16; i++){
			T delta  = Tbeedt[i] - Tbeedt2[i];
			gradc[i] = delta / (2.0*TEST_FINITE_DIFF_EPSILON);
		}
	}
}
  
     
template <typename T>
__host__
void analyticalTbdt(T *x, T *grad){
	T s_cosq[NUM_POS];         	T s_sinq[NUM_POS];      	
	T s_Tb[36*NUM_POS];			T d_Tb[36*NUM_POS]; 	   	initT<T>(d_Tb); 
	T s_Tb_dx[32*NUM_POS];	   	T s_TbTdt[32*NUM_POS];		T s_Tb_dt_dx[16*2*NUM_POS];
    load_Tbdtdx<T>(x,s_Tb,d_Tb,s_sinq,s_cosq,s_Tb_dx,s_TbTdt,s_Tb_dt_dx);
   	for (int k = 0; k < STATE_SIZE; k++){
		for (int i = 0; i < 16; i++){grad[16*k + i] = s_Tb_dt_dx[16*k + i];}
	}
}

template <typename T>
__host__
void finiteDiffTdt(T *x, T *grad, int ld_grad){
	T s_x[2*STATE_SIZE];	T s_cosq[NUM_POS];		T s_sinq[NUM_POS];
	T s_Tb[36*NUM_POS];		T d_Tb[36*NUM_POS];		initT<T>(d_Tb);			
	T s_TbTdt[36*NUM_POS];	T s_TbTdt2[36*NUM_POS];	T s_T[36*NUM_POS];
	#pragma unroll
	for (int diff_ind = 0; diff_ind < STATE_SIZE; diff_ind++){
		T *gradc = &grad[ld_grad*diff_ind];
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? TEST_FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
		}
		// compute Tdt
		load_Tbdt<T>(s_x,			  s_Tb,d_Tb,s_cosq,s_sinq,s_TbTdt);		compute_T_TA_J<T>(s_Tb,s_T,nullptr,nullptr,s_TbTdt);
		load_Tbdt<T>(&s_x[STATE_SIZE],s_Tb,d_Tb,s_cosq,s_sinq,s_TbTdt2);	compute_T_TA_J<T>(s_Tb,s_T,nullptr,nullptr,s_TbTdt2);
		T *Teedt = &s_TbTdt[16*(NUM_POS*2-1)];
		T *Teedt2 = &s_TbTdt2[16*(NUM_POS*2-1)];
		// now do finite diff rule
		#pragma unroll
		for (int i = 0; i < 16; i++){
			T delta  = Teedt[i] - Teedt2[i];
			gradc[i] = delta / (2.0*TEST_FINITE_DIFF_EPSILON);
		}
	}
}
  
     
template <typename T>
__host__
void analyticalTdt(T *x, T *grad){
	T s_cosq[NUM_POS];         	T s_sinq[NUM_POS];      		T s_temp1[32*NUM_POS];
	T s_Tb[36*NUM_POS];			T d_Tb[36*NUM_POS]; 	   		initT<T>(d_Tb); 
	T s_Tb_dx[16*NUM_POS];	   	T s_TbTdt[32*NUM_POS];			T s_T[36*NUM_POS];
	T s_Tb_dt_dx[32*NUM_POS];	T s_T_dt_dx[32*NUM_POS];		T s_T_dt_dx_prev[32*NUM_POS];
   	// load in Tb, Tb_dx, Tb_dt
    load_Tbdtdx<T>(x,s_Tb,d_Tb,s_sinq,s_cosq,s_Tb_dx,s_TbTdt,s_Tb_dt_dx);
    compute_T_TA_J<T>(s_Tb,s_T,nullptr,nullptr,s_TbTdt);
    T *s_Tb_dt = s_TbTdt;   T *s_T_dt = &s_TbTdt[16*NUM_POS];
    T *s_T_dx = s_temp1; 	T *s_T_dx_prev = &s_T_dx[16*NUM_POS];
    compute_T_dtdx<T>(s_Tb,s_Tb_dx,s_Tb_dt,s_Tb_dt_dx,s_T,s_T_dx,s_T_dt,s_T_dt_dx,s_T_dx_prev,s_T_dt_dx_prev);
   	for (int k = 0; k < STATE_SIZE; k++){
		for (int i = 0; i < 16; i++){grad[16*k + i] = s_T_dt_dx[16*k + i];}
	}
}

template <typename T>
__host__
void finiteDiffPos(T *x, T *eePosGrad, int ld_grad){
	T s_x[2*STATE_SIZE];	T eePos[2*6];
	#pragma unroll
	for (int diff_ind = 0; diff_ind < NUM_POS; diff_ind++){
		T *eeGrad = &eePosGrad[ld_grad*diff_ind];
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? TEST_FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
		}
		// compute eePos
		compute_eePos_scratch<T>( s_x, 			    eePos);
		compute_eePos_scratch<T>(&s_x[STATE_SIZE], &eePos[6]);
		// now do finite diff rule
		#pragma unroll
		for (int i = 0; i < 6; i++){
			T deltaPos = eePos[i] - eePos[i+6];
			eeGrad[i]  = deltaPos / (2.0*TEST_FINITE_DIFF_EPSILON);
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
			T adj = (diff_ind == i ? TEST_FINITE_DIFF_EPSILON : 0.0);
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
			eeGrad[i]   = deltaPos / (2.0*TEST_FINITE_DIFF_EPSILON);
			eeGrad[i+6] = deltaVel / (2.0*TEST_FINITE_DIFF_EPSILON);
		}
	}
}

template <typename T>
__host__
void analyticalVel(T *x, T *eePosVelGrad, int ld_grad){
	T s_cosq[NUM_POS];         	T s_sinq[NUM_POS];      	T s_Tb[36*NUM_POS];
	T d_Tb[36*NUM_POS]; 	   	initT<T>(d_Tb); 			T s_Tb_dt_dx[16*NUM_POS];
	T s_Tb_dx[32*NUM_POS];	   	T s_TbTdt[32*NUM_POS];		T s_T[36*NUM_POS];
	T s_temp1[32*NUM_POS];	   	T s_temp2[32*NUM_POS];		T s_temp3[32*NUM_POS];
	T s_eePos[6];				T s_eeVel[6];
 	compute_eePosVel_dx<T>(x, s_Tb, d_Tb, s_cosq, s_sinq, s_Tb_dx, s_TbTdt, s_Tb_dt_dx, s_T, 
                           s_temp1, s_temp2, s_temp3, s_eePos, s_eeVel, eePosVelGrad, ld_grad);
}

template <typename T>
__host__
void finiteDiffCost(T *x, T *u, T *goal, T *grad){
	T s_x[2*STATE_SIZE];	T s_u[2*CONTROL_SIZE];	T cost, cost2;
	#if EE_COST
		T eePos[2*6];	T eeVel[2*6];
	#endif
	#pragma unroll
	for (int diff_ind = 0; diff_ind < STATE_SIZE+CONTROL_SIZE; diff_ind++){
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? TEST_FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
			if (i < CONTROL_SIZE){
				T val = u[i];
				T adj = (diff_ind == i + STATE_SIZE ? TEST_FINITE_DIFF_EPSILON : 0.0);
				s_u[i] 				= val + adj;
				s_u[i + CONTROL_SIZE] = val - adj;
			}
		}
		#if EE_COST
			// compute eePos and Vel
			compute_eeVel_scratch<T>(s_x, 			   eePos, 	  eeVel);
			compute_eeVel_scratch<T>(&s_x[STATE_SIZE], &eePos[6], &eeVel[6]);
			// then compute costs
			cost = costFunc(  eePos,    goal,  s_x, 			   s_u, 			  0, eeVel);
			cost2 = costFunc(&eePos[6], goal, &s_x[STATE_SIZE], &s_u[CONTROL_SIZE], 0, &eeVel[6]);
		#else
			cost = costFunc(  s_x, 				s_u, 			   goal, 0);
			cost2 = costFunc(&s_x[STATE_SIZE], &s_u[CONTROL_SIZE], goal, 0);
		#endif
		// now do finite diff rule
		grad[diff_ind] = (cost - cost2) / (2.0*TEST_FINITE_DIFF_EPSILON);
	}
}

template <typename T>
__host__
void analyticalCost(T *x, T *u, T *goal, T *grad){
	// space for Hessian
	int ld_H = STATE_SIZE+CONTROL_SIZE; 	T Hk[ld_H*ld_H];
	#if EE_COST
		// first compute eePos and eeVel and grads
		T s_eePos[6];	T s_eeVel[6]; 	T s_deePos[6*STATE_SIZE];	T s_deeVel[12*STATE_SIZE];
		compute_eeVel_scratch<T>(x,s_eePos,s_eeVel);	
		analyticalPos<T>(x,s_deePos);	analyticalVel<T>(x,s_deeVel,12);
		T *JT = nullptr;
		costGrad<T>(Hk,grad,s_eePos,s_deePos,goal,x,u,0,ld_H,JT,-1,s_eeVel,s_deeVel);
	#else
		costGrad(Hk,grad,x,u,goal,0,ld_H);
	#endif
}

template <typename T>
__host__
void loadAndClearPos(T *x, T *grad, T *grad2, int gradSize, T *u = nullptr, T *goal = nullptr){
	// load random state
	#pragma unroll
	for (int i=0; i < NUM_POS; i++){
		x[i] = static_cast<T>(randDistq(randEng));
		x[i+NUM_POS] = static_cast<T>(randDistqd(randEng));
		if (u != nullptr){u[i] = static_cast<T>(randDistu(randEng));}
		if (goal != nullptr){
			#if EE_COST
				if(i < 6){goal[i] = static_cast<T>(randDistg(randEng));}
			#else
				goal[i] = static_cast<T>(randDistq(randEng));
				goal[i+NUM_POS] = static_cast<T>(randDistqd(randEng));
			#endif
		}
	}
	// clear grads
	for (int i=0; i < gradSize; i++){
		grad[i] = 0;	grad2[i] = 0;
	}
}

template <typename T>
__host__
void testT(int runv2 = 0){
	// allocate
	int ld_grad = 16;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*NUM_POS*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*NUM_POS*sizeof(T));

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,ld_grad*NUM_POS);
		// compute
		if (runv2){analyticalT2<T>(x,grad);}
		else{analyticalT<T>(x,grad);}
		finiteDiffT<T>(x,grad2,ld_grad);
		// compare
		#pragma unroll
		for (int c = 0; c < NUM_POS; c++){
			#pragma unroll
			for (int r = 0; r < 16; r++){
				int ind = c*ld_grad + r;
				T delta = abs(grad[ind] - grad2[ind]);
				T err = abs(grad2[ind] == 0 ? (grad[ind] == 0 ? 0 : delta/grad[ind]*100) : delta/grad2[ind]*100);
				if (err > ERR_TOL){
					printf("rep[%d] ind[%d]=c,r[%d,%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,c,r,err,grad[ind],grad2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2);
}

template <typename T>
__host__
void testTbdt(){
	// allocate
	int ld_grad = 16;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,ld_grad*STATE_SIZE);
		// compute
		analyticalTbdt<T>(x,grad);
		finiteDiffTbdt<T>(x,grad2,ld_grad);
		// compare
		#pragma unroll
		for (int c = 0; c < STATE_SIZE; c++){
			#pragma unroll
			for (int r = 0; r < 16; r++){
				int ind = c*ld_grad + r;
				T delta = abs(grad[ind] - grad2[ind]);
				T err = abs(grad2[ind] == 0 ? (grad[ind] == 0 ? 0 : delta/grad[ind]*100) : delta/grad2[ind]*100);
				if (err > ERR_TOL){
					printf("rep[%d] ind[%d]=c,r[%d,%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,c,r,err,grad[ind],grad2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2);
}

template <typename T>
__host__
void testTdt(){
	// allocate
	int ld_grad = 16;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,ld_grad*STATE_SIZE);
		// compute
		analyticalTdt<T>(x,grad);
		finiteDiffTdt<T>(x,grad2,ld_grad);
		// compare
		#pragma unroll
		for (int c = 0; c < STATE_SIZE; c++){
			#pragma unroll
			for (int r = 0; r < 16; r++){
				int ind = c*ld_grad + r;
				T delta = abs(grad[ind] - grad2[ind]);
				T err = abs(grad2[ind] == 0 ? (grad[ind] == 0 ? 0 : delta/grad[ind]*100) : delta/grad2[ind]*100);
				if (err > ERR_TOL){
					printf("rep[%d] ind[%d]=c,r[%d,%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,c,r,err,grad[ind],grad2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2);
}

template <typename T>
__host__
void testPos(){
	// allocate
	int ld_grad = 6;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*NUM_POS*sizeof(T));

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,ld_grad*NUM_POS);
		// compute
		analyticalPos<T>(x,grad);
		finiteDiffPos<T>(x,grad2,ld_grad);
		// compare
		#pragma unroll
		for (int c = 0; c < NUM_POS; c++){
			#pragma unroll
			for (int r = 0; r < 6; r++){
				int ind = c*ld_grad + r;
				T delta = abs(grad[ind] - grad2[ind]);
				T err = abs(grad2[ind] == 0 ? (grad[ind] == 0 ? 0 : delta/grad[ind]*100) : delta/grad2[ind]*100);
				if (err > ERR_TOL){
					printf("rep[%d] ind[%d]=c,r[%d,%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,c,r,err,grad[ind],grad2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2);
}

template <typename T>
__host__
void testVel(){
	// allocate
	int ld_grad = 12;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,ld_grad*STATE_SIZE);
		// compute
		analyticalVel<T>(x,grad,ld_grad);
		finiteDiffVel<T>(x,grad2,ld_grad);
		// compare
		#pragma unroll
		for (int c = 0; c < STATE_SIZE; c++){
			#pragma unroll
			for (int r = 0; r < 12; r++){
				int ind = c*ld_grad + r;
				T delta = abs(grad[ind] - grad2[ind]);
				T err = abs(grad2[ind] == 0 ? (grad[ind] == 0 ? 0 : delta/grad[ind]*100) : delta/grad2[ind]*100);
				if (err > ERR_TOL){
					printf("rep[%d] ind[%d]=c,r[%d,%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,c,r,err,grad[ind],grad2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2);
}

template <typename T>
__host__
void testCost(){
	// allocate
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *u =     (T *)malloc(CONTROL_SIZE*sizeof(T));
	T *grad =  (T *)malloc((STATE_SIZE+CONTROL_SIZE)*sizeof(T));	
	T *grad2 = (T *)malloc((STATE_SIZE+CONTROL_SIZE)*sizeof(T));	
	T *goal;
	#if EE_COST
		goal =(T *)malloc(6*sizeof(T));
	#else
		goal =(T *)malloc(STATE_SIZE*sizeof(T));
	#endif

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,STATE_SIZE+CONTROL_SIZE,u,goal);
		// compute
		analyticalCost<T>(x,u,goal,grad);
		finiteDiffCost<T>(x,u,goal,grad2);
		// compare
		#pragma unroll
		for (int ind = 0; ind < STATE_SIZE + CONTROL_SIZE; ind++){
			T delta = abs(grad[ind] - grad2[ind]);
			T err = abs(grad2[ind] == 0 ? (grad[ind] == 0 ? 0 : delta/grad[ind]*100) : delta/grad2[ind]*100);
			if (err > ERR_TOL){
				printf("rep[%d] ind[%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,err,grad[ind],grad2[ind]);
			}
		}
	}
	//free
	free(x); free(u); free(grad); free(grad2); free(goal);
}


template <typename T>
__host__
void compareEEPosdq(){
	// allocate
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(12*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(12*STATE_SIZE*sizeof(T));	

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,12*STATE_SIZE);
		// compute
		analyticalPos<T>(x,grad);
		analyticalVel<T>(x,grad2,12);
		// compare
		#pragma unroll
		for (int c = 0; c < STATE_SIZE; c++){
			#pragma unroll
			for (int r = 0; r < 6; r++){
				int ind_1 = c*6 + r;	int ind_2 = c*12 + r;
				T delta = abs(grad[ind_1] - grad2[ind_2]);
				T err = abs(grad2[ind_2] == 0 ? (grad[ind_1] == 0 ? 0 : delta/grad[ind_1]*100) : delta/grad2[ind_2]*100);
				if (err > ERR_TOL){
					printf("rep[%d] c,r[%d,%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,c,r,err,grad[ind_1],grad2[ind_2]);
				}
			}
		}
	}
	//free
	free(x); free(grad); free(grad2);
}

template <typename T>
__host__
void compareCostdq(){
	// allocate
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *u =     (T *)malloc(CONTROL_SIZE*sizeof(T));
	T *grad =  (T *)malloc((STATE_SIZE+CONTROL_SIZE)*sizeof(T));	
	T *grad2 = (T *)malloc((STATE_SIZE+CONTROL_SIZE)*sizeof(T));	
	T *goal;
	#if EE_COST
		goal =(T *)malloc(6*sizeof(T));
	#else
		goal =(T *)malloc(STATE_SIZE*sizeof(T));
	#endif

	int ld_H    = STATE_SIZE + CONTROL_SIZE;
	T *Hessian  = (T *)malloc((STATE_SIZE+CONTROL_SIZE)*ld_H*sizeof(T));
	T *Hessian2 = (T *)malloc((STATE_SIZE+CONTROL_SIZE)*ld_H*sizeof(T));
	T Tbody[36*NUM_POS]; 	   	initT<T>(Tbody);

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,STATE_SIZE+CONTROL_SIZE,u,goal);
		// compute
		T s_sinq[NUM_POS];			T s_cosq[NUM_POS];
		T s_Tb[36*NUM_POS];			T s_Tb_dx[36*NUM_POS];
		T s_T[36*NUM_POS];			T s_T_dx[36*NUM_POS];
		T s_eePos[6];				T s_deePos[6*NUM_POS];
		T s_TbTdt[32*NUM_POS]; 		T s_Tb_dt_dx[16*NUM_POS];
		T s_eeVel[6];		  		T s_deePosVel[12*NUM_POS];
		T s_T_dt_dx[32*NUM_POS];  	T s_T_dt_dx_p[32*NUM_POS];
		
		compute_eePosVel_dx<T>(x, s_Tb, Tbody, s_cosq, s_sinq, s_Tb_dx, s_TbTdt, s_Tb_dt_dx, s_T, 
                               s_T_dx, s_T_dt_dx, s_T_dt_dx_p, s_eePos, s_eeVel, s_deePosVel, 12);
		costGrad<T>(Hessian,grad,s_eePos,s_deePos,goal,x,u,0,ld_H,nullptr,0,s_eeVel,s_deePosVel);

		compute_eePos<T>(s_T,s_eePos,s_T_dx,s_deePos,s_sinq,s_Tb,s_Tb_dx,x,s_cosq,Tbody);
		costGrad<T>(Hessian2,grad2,s_eePos,s_deePos,goal,x,u,0,ld_H);
		
		// compare
		#pragma unroll
		for (int ind = 0; ind < STATE_SIZE + CONTROL_SIZE; ind++){
			T delta = abs(grad[ind] - grad2[ind]);
			T err = abs(grad2[ind] == 0 ? (grad[ind] == 0 ? 0 : delta/grad[ind]*100) : delta/grad2[ind]*100);
			if (err > ERR_TOL){
				printf("rep[%d] Gradient ind[%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,ind,err,grad[ind],grad2[ind]);
			}
		}
		#pragma unroll
		for (int c = 0; c < NUM_POS; c++){
			#pragma unroll
			for (int r = 0; r < NUM_POS; r++){
				int ind = c*ld_H + r;
				T delta = abs(Hessian[ind] - Hessian2[ind]);
				T err = abs(Hessian2[ind] == 0 ? (Hessian[ind] == 0 ? 0 : delta/Hessian[ind]*100) : delta/Hessian2[ind]*100);
				if (err > ERR_TOL){
					printf("rep[%d] Hessian c,r[%d,%d] has err[%.2f] percent for analytical[%.8f] vs finiteDiff[%.8f]\n",rep,c,r,err,Hessian[ind],Hessian2[ind]);
				}
			}
		}
	}
	//free
	free(x); free(u); free(grad); free(grad2); free(goal);
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	char mode = '?';
	if (argc > 1){mode = argv[1][0];}
	switch (mode){
		case 'T':
			testT<algType>(argv[1][1]);
			break;
		case 'b':
			testTbdt<algType>();
			break;
		case 'd':
			testTdt<algType>();
			break;
		case 'P':
			testPos<algType>();
			break;
		case 'V':
			testVel<algType>();
			break;
		case 'C':
			testCost<algType>();
			break;
		case 'c':
			compareEEPosdq<algType>();
			break;
		case 'm':
			compareCostdq<algType>();
			break;
		default:
			printf("Input is [P]os, [V]el, [T]ransforms, T[b]ody/dT, Transform/[d]T, full [C]ost, or [c]ompare Pdq methods or costGrad [m]ethods\n");
			return 1;
	}
	return 0;
}

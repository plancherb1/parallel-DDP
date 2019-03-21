/*******
nvcc -std=c++11 -o testCostGrad.exe testCostGrad.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -gencode arch=compute_52,code=sm_52 -rdc=true -O3
*******/

#define PLANT 4
#define EE_COST 1
#include "../DDPHelpers.cuh"
#include <random>
#define NUM_REPS 1
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
			T adj = (diff_ind == i ? FINITE_DIFF_EPSILON : 0.0);
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
			gradc[i] = delta / (2.0*FINITE_DIFF_EPSILON);
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
			T adj = (diff_ind == i ? FINITE_DIFF_EPSILON : 0.0);
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
			gradc[i] = delta / (2.0*FINITE_DIFF_EPSILON);
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
			T adj = (diff_ind == i ? FINITE_DIFF_EPSILON : 0.0);
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
			gradc[i] = delta / (2.0*FINITE_DIFF_EPSILON);
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
			T adj = (diff_ind == i ? FINITE_DIFF_EPSILON : 0.0);
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
			eeGrad[i]  = deltaPos / (2.0*FINITE_DIFF_EPSILON);
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
	T d_Tb[36*NUM_POS]; 	   	initT<T>(d_Tb); 			T s_Tb_dt_dx[16*NUM_POS];
	T s_Tb_dx[32*NUM_POS];	   	T s_TbTdt[32*NUM_POS];		T s_T[36*NUM_POS];
	T s_temp1[32*NUM_POS];	   	T s_temp2[32*NUM_POS];		T s_temp3[32*NUM_POS];
	T s_eePos[6];				T s_eeVel[6];
 	compute_eePosVel_dx<T>(x, s_Tb, d_Tb, s_cosq, s_sinq, s_Tb_dx, s_TbTdt, s_Tb_dt_dx, s_T, 
                           s_temp1, s_temp2, s_temp3, s_eePos, s_eeVel, eePosVelGrad, ld_grad);
}

template <typename T>
__host__
void finiteDiffCost(T *x, T *u, T *eeGoal, T *grad){
	T s_x[2*STATE_SIZE];	T s_u[2*CONTROL_SIZE];	T eePos[2*6];	T eeVel[2*6];
	#pragma unroll
	for (int diff_ind = 0; diff_ind < STATE_SIZE+CONTROL_SIZE; diff_ind++){
		#pragma unroll
		for (int i = 0; i < STATE_SIZE; i++){
			T val = x[i];	
			T adj = (diff_ind == i ? FINITE_DIFF_EPSILON : 0.0);
			s_x[i] 				= val + adj;
			s_x[i + STATE_SIZE] = val - adj;
			if (i < CONTROL_SIZE){
				T val = u[i];
				T adj = (diff_ind == i + STATE_SIZE ? FINITE_DIFF_EPSILON : 0.0);
				s_u[i] 				= val + adj;
				s_u[i + CONTROL_SIZE] = val - adj;
			}
		}
		// compute eePos and Vel
		compute_eeVel_scratch<T>(s_x, 			   eePos, 	  eeVel);
		compute_eeVel_scratch<T>(&s_x[STATE_SIZE], &eePos[6], &eeVel[6]);
		// then compute costs
		T cost = costFunc(eePos,      eeGoal, s_x, 				s_u, 				0, eeVel);
		T cost2 = costFunc(&eePos[6], eeGoal, &s_x[STATE_SIZE], &s_u[CONTROL_SIZE], 0, &eeVel[6]);
		// now do finite diff rule
		grad[diff_ind] = (cost - cost2) / (2.0*FINITE_DIFF_EPSILON);
	}
}

template <typename T>
__host__
void analyticalCost(T *x, T *u, T *eeGoal, T *grad){
	// space for Hessian
	int ld_H = STATE_SIZE+CONTROL_SIZE; 	T Hk[ld_H*ld_H];
	// first compute eePos and eeVel and grads
	T s_eePos[6];	T s_eeVel[6]; 	T s_deePos[6*STATE_SIZE];	T s_deeVel[12*STATE_SIZE];
	compute_eeVel_scratch<T>(x,s_eePos,s_eeVel);	
	analyticalPos<T>(x,s_deePos);	analyticalVel<T>(x,s_deeVel,12);
	T *JT = nullptr;
	costGrad<T>(Hk,grad,s_eePos,s_deePos,eeGoal,x,u,0,ld_H,JT,-1,s_eeVel,s_deeVel);

	// int k = 0; T *d_JT = nullptr; int tid = -1; T *gk = costGrad; T *s_x = x; T *s_u = u; T *d_eeGoal = eeGoal;
	// // then compute the gradient
	// int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
	// int start, delta; singleLoopVals(&start,&delta);
	// #pragma unroll
	// for (int r = start; r < DIM_g_r; r += delta){
	//   	T val = 0.0;
	//   	if (r < NUM_POS){val += deeCost<T>(s_eePos,s_deePos,d_eeGoal,k,r,s_eeVel,s_deeVel);}
	//   	// add on the joint level state cost (tend to zero regularizer) and control cost
	//   	if (r < NUM_POS){val += (k == NUM_TIME_STEPS - 1 ? QF_xHAND : Q_xHAND)*s_x[r];}
	//   	else if (r < STATE_SIZE){val += (k == NUM_TIME_STEPS - 1 ? QF_xdHAND : Q_xdHAND)*s_x[r];}
	//   	else{val += (k == NUM_TIME_STEPS - 1 ? 0 : R_HAND)*s_u[r-STATE_SIZE];}
	//   	// add on any limit costs if needed
	//   	#if USE_LIMITS_FLAG
 //      		val += limitCosts<T,1>(s_x,s_u,r,k);
 //  		#endif
	//   	gk[r] = val;
	// }
	// hd__syncthreads();
	// // d2J/dx2 = H \approx dh_i/dx'*dh_i/dx + other stuff
	// #pragma unroll
	// for (int c = starty; c < DIM_H_c; c += dy){
	//   	T *H = &Hk[c*ld_H];
	//   	#pragma unroll
	//   	for (int r= startx; r<DIM_H_r; r += dx){
	//      	T val = 0.0;
	//      	// multiply two columns for pseudo-Hessian (dropping d2q/dh2 term)
	//      	if (c < NUM_POS && r < NUM_POS){
	//         	#pragma unroll
	//         	for (int j = 0; j < 6; j++){
	//            		val += s_deePos[r*6+j]*s_deePos[c*6+j];
	//            		if (s_deeVel != nullptr){val += s_deeVel[r*12+6+j]*s_deeVel[c*12+6+j];}
	//         	}
	//      	}
	// 	    // if applicable add on the joint level state cost (tend to zero regularizer) and control cost
	// 	    if (r == c){
	//         	if (r < NUM_POS){val += (k == NUM_TIME_STEPS - 1 ? QF_xHAND : Q_xHAND);}
	//         	else if (r < STATE_SIZE){val += (k == NUM_TIME_STEPS - 1 ? QF_xdHAND : Q_xdHAND);}
	//         	else {val += (k== NUM_TIME_STEPS - 1) ? 0.0 : R_HAND;}
	//         	// add on any limit costs if needed
	//         	#if USE_LIMITS_FLAG
 //      				val += limitCosts<T,2>(s_x,s_u,r,k);
 //  				#endif
	//      	}
	//      	H[r] = val;//s_g[k]*s_g[i]; // before we multiplied gradient but that isn't correct
	//   	}
	// }
	// //if cost asked for compute it
	// bool flag = d_JT != nullptr; int ind = (tid != -1 ? tid : k);
	// #ifdef __CUDA_ARCH__
	// 	if(threadIdx.x != 0 || threadIdx.y != 0){flag = 0;}
	// 	if (flag){d_JT[ind] = costFunc(s_eePos,d_eeGoal,s_x,s_u,k);}
	// 		// printf("Cost for ind[%d] for k[%d] and eePos[%f %f %f %f %f %f] and goal[%f %f %f %f %f %f] with u[%f %f %f %f %f %f] and x[%f %f %f %f %f %f %f %f %f %f %f %f] is J[%f]\n",
	// 		// 	    ind,k,s_eePos[0],s_eePos[1],s_eePos[2],s_eePos[3],s_eePos[4],s_eePos[5],d_eeGoal[0],d_eeGoal[1],d_eeGoal[2],d_eeGoal[3],d_eeGoal[4],d_eeGoal[5],
	// 		// 	    s_u[0],s_u[1],s_u[2],s_u[3],s_u[4],s_u[5],s_u[6],
	// 		// 	    s_x[0],s_x[1],s_x[2],s_x[3],s_x[4],s_x[5],s_x[6],s_x[7],s_x[8],s_x[9],s_x[10],s_x[11],s_x[12],s_x[13],d_JT[ind]);}
	// #else
	// 	if (flag){d_JT[ind] += costFunc(s_eePos,d_eeGoal,s_x,s_u,k);}
	// #endif
}

template <typename T>
__host__
void loadAndClearPos(T *x, T *grad, T *grad2, int gradSize, T *u = nullptr, T *eeGoal = nullptr){
	// load random state
	#pragma unroll
	for (int i=0; i < NUM_POS; i++){
		x[i] = static_cast<T>(randDistq(randEng));
		x[i+NUM_POS] = static_cast<T>(randDistqd(randEng));
		if (u != nullptr){u[i] = static_cast<T>(randDistu(randEng));}
		if (eeGoal != nullptr && i < 6){eeGoal[i] = static_cast<T>(randDistg(randEng));}
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
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);

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
	free(x); free(grad); free(grad2); free(Tbody);
}

template <typename T>
__host__
void testTbdt(){
	// allocate
	int ld_grad = 16;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);

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
	free(x); free(grad); free(grad2); free(Tbody);
}

template <typename T>
__host__
void testTdt(){
	// allocate
	int ld_grad = 16;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);

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
	free(x); free(grad); free(grad2); free(Tbody);
}

template <typename T>
__host__
void testPos(){
	// allocate
	int ld_grad = 6;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*NUM_POS*sizeof(T));
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);

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
	free(x); free(grad); free(grad2); free(Tbody);
}

template <typename T>
__host__
void testVel(){
	// allocate
	int ld_grad = 12;
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *grad =  (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));	
	T *grad2 = (T *)malloc(ld_grad*STATE_SIZE*sizeof(T));
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);

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
	free(x); free(grad); free(grad2); free(Tbody);
}

template <typename T>
__host__
void testCost(){
	// allocate
	T *x =     (T *)malloc(STATE_SIZE*sizeof(T));
	T *u =     (T *)malloc(CONTROL_SIZE*sizeof(T));
	T *grad =  (T *)malloc((STATE_SIZE+CONTROL_SIZE)*sizeof(T));	
	T *grad2 = (T *)malloc((STATE_SIZE+CONTROL_SIZE)*sizeof(T));	
	T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);
	T *eeGoal =(T *)malloc(6*sizeof(T));

	// compare for NUM_REPS
	for (int rep = 0; rep < NUM_REPS; rep++){
		// relod and clear
		loadAndClearPos<T>(x,grad,grad2,STATE_SIZE+CONTROL_SIZE,u,eeGoal);
		// compute
		analyticalCost<T>(x,u,eeGoal,grad);
		finiteDiffCost<T>(x,u,eeGoal,grad2);
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
	free(x); free(u); free(grad); free(grad2); free(Tbody); free(eeGoal);
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
		default:
			printf("Input is [P]os, [V]el, [T]ransforms, T[b]ody/dT, Transform/[d]T or full [C]ost\n");
			return 1;
	}
	return 0;
}

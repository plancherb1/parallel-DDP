/*****************************************************************
 * Cart Dynamics
 *
 * To make sure we don't have compile errors we need dummy funcs:
 * initI(T *s_I), initT(T *s_T), compute_eePos(T *s_T, T *s_eePos, T *s_dT, T *s_deePos, T *s_sinq, T *s_Tb, T *s_dTb, T *s_x, T *s_cosq, T *d_Tbody);
 *
 * dynamics(T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody, T *s_eePos = nullptr, int reps = 1);
 * 
 * dynamicsGradient(T *s_dqdd, T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody);
 *****************************************************************/

// dynamics parameters
#define GRAVITY -9.81
#define PI 3.1416
#define M_CART 10
#define M_POLE 1
#define L_POLE 0.5
#define ML_POLE (M_POLE * L_POLE)
#define MLL_POLE (ML_POLE * L_POLE)

// To make sure we don't have compile errors we need dummy funcs:
template <typename T> __host__ __device__ __forceinline__ void initI(T *s_I){return;} 
template <typename T> __host__ __device__ __forceinline__ void initT(T *s_T){return;}
template <typename T> __host__ __device__ __forceinline__ void compute_eePos(T *s_eePos, T *s_T, T *s_Tb, T *s_sinq, T *s_cosq, T *s_x, T *d_Tbody){return;}
template <typename T> __host__ __device__ __forceinline__ void compute_eePos(T *s_T, T *s_eePos, T *s_dT, T *s_deePos, T *s_sinq, T *s_Tb, T *s_dTb, T *s_x, T *s_cosq, T *d_Tbody){return;}

// Actual funcs we'll use (no EE or RBDYN needed)
template <typename T>
__host__ __device__ __forceinline__
void dynamics(T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody, T *s_eePos = nullptr, int reps = 1){
	int start, delta; singleLoopVals(&start,&delta);
   	for(int iter = start; iter < reps; iter += delta){
      	T *s_xk = &s_x[STATE_SIZE*iter];
      	T *s_uk = &s_u[NUM_POS*iter];
      	T *s_qddk = &s_qdd[NUM_POS*iter];
      	s_qddk[0] = s_uk[0] + GRAVITY*sin(s_xk[0]);
   	}
}

template <typename T>
__host__ __device__ __forceinline__
void dynamicsGradient(T *s_dqdd, T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody){
	#ifdef __CUDA_ARCH__
      	if (threadIdx.x != 0 || threadIdx.y != 0){return;}
	#endif
    // if s_qdd requested then return the value
    if (s_qdd != nullptr){dynamics(s_qdd,s_x,s_u,d_I,d_Tbody);}
    // then return gradient
	s_dqdd[0] = GRAVITY*cos(s_x[0]); //dx
	s_dqdd[1] = 0.0; //dxd
	s_dqdd[2] = 1;   //du
}
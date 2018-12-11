/*****************************************************************
 * Quad Cost Funcs
 *
 * TBD NEED TO ADD DOC HERE
 *****************************************************************/

// cost parameters -- note no end effector so end effector mode makes no sense
#if EE_COST
	#error "Quad does not have an end effector -- please compile with EE_COST turned off."
#else
 	// EE_COST func definitons so no compile error since we know we'll never take this path.
	template <typename T> __host__ __device__ __forceinline__
	void costFunc(T *s_cost, T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k){return;}
 	template <typename T> __host__ __device__ __forceinline__
 	T costFunc(T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k){return;}
 	template <typename T> __host__ __device__ __forceinline__
	void costGrad(T *Hk, T*gk, T *s_eePos, T *s_deePos, T *d_eeGoal, T *s_x, T *s_u, int k, int ld_H, T *d_JT = nullptr, int tid = -1){return;}
#endif
#define Q1 0.01 // xyz
#define Q2 0.001 // rpy
#define Q3 2.0 // xyzdot
#define Q4 2.0 // rpydot
#define R  5.0
#define QF 1000.0
#define QR(i) (i < 3 ? Q1 : (i < 6 ? Q2 : (i < 9 ? Q3 : (i < 12 ? Q4 : R))))

// joint level cost func
template <typename T>
__host__ __device__ __forceinline__
T costFunc(T *xk, T *uk, T *xgk, int k){
	T cost = 0.0;
	#pragma unroll
    for (int i=0; i<STATE_SIZE; i++){cost += (T) (k == NUM_TIME_STEPS - 1 ? QF : QR(i))*pow(xk[i]-xgk[i],2);}
	if (k != NUM_TIME_STEPS - 1){
		#pragma unroll
        for (int i=0; i<CONTROL_SIZE; i++){cost += (T) R*pow(uk[i],2);}
    }
	return 0.5*cost;
}

// joint level cost grad
template <typename T>
__host__ __device__ __forceinline__
void costGrad(T *Hk, T *gk, T *xk, T *uk, T *xgk, int k, int ld_H){
  	#pragma unroll
  	for (int i=0; i<STATE_SIZE+CONTROL_SIZE; i++){
  		#pragma unroll
     	for (int j=0; j<STATE_SIZE+CONTROL_SIZE; j++){
        	Hk[i*ld_H + j] = (T) (i != j ? 0.0 : (k == NUM_TIME_STEPS - 1 ? (i < STATE_SIZE ? QF : 0.0) : QR(i)));
     	}  
  	}
  	#pragma unroll
  	for (int i=0; i<STATE_SIZE; i++){gk[i] = (T) (k == NUM_TIME_STEPS - 1 ? QF : QR(i))*(xk[i]-xgk[i]);}
  	#pragma unroll
  	for (int i=0; i<CONTROL_SIZE; i++){gk[i+STATE_SIZE] = (T) (k == NUM_TIME_STEPS - 1 ? 0.0 : R)*uk[i];}
}
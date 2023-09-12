/*****************************************************************
* Cart Cost Funcs
*
* TBD NEED TO ADD DOC HERE
*****************************************************************/

// cost parameters -- note no end effector so end effector mode makes no sense
#if EE_COST_PDDP
    #error "Pend does not have an end effector -- please compile with EE_COST_PDDP turned off."
#else
    // EE_COST_PDDP func definitons so no compile error since we know we'll never take this path.
    template <typename T> __host__ __device__ __forceinline__
    void costFunc(T *s_cost, T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k){return;}
    template <typename T> __host__ __device__ __forceinline__
    T costFunc(T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k){return;}
    template <typename T> __host__ __device__ __forceinline__
    void costGrad(T *Hk, T*gk, T *s_eePos, T *s_deePos, T *d_eeGoal, T *s_x, T *s_u, int k, int ld_H, T *d_JT = nullptr, int tid = -1){return;}
#endif
#define PI 3.1416
#define Q1 1.0
#define Q2 0.1
#define R 0.1
#define QF 1000.0
#define QR(i) (i == 0 ? Q1 : (i == 2 ? Q2 : R))

// joint level cost func
template <typename T>
__host__ __device__ __forceinline__
T costFunc(T *xk, T *uk, T *xgk, int k){
    T cost = 0.0;
    #pragma unroll
    for (int i=0; i<STATE_SIZE_PDDP; i++){cost += (T) (k == NUM_TIME_STEPS - 1 ? QF : QR(i))*pow(xk[i]-xgk[i],2);}
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
    for (int i=0; i<STATE_SIZE_PDDP+CONTROL_SIZE; i++){
        #pragma unroll
        for (int j=0; j<STATE_SIZE_PDDP+CONTROL_SIZE; j++){
            Hk[i*ld_H + j] = (T) (i != j ? 0.0 : (k == NUM_TIME_STEPS - 1 ? (i < STATE_SIZE_PDDP ? QF : 0.0) : QR(i)));
        }  
    }
    #pragma unroll
    for (int i=0; i<STATE_SIZE_PDDP; i++){gk[i] = (T) (k == NUM_TIME_STEPS - 1 ? QF : QR(i))*(xk[i]-xgk[i]);}
    #pragma unroll
    for (int i=0; i<CONTROL_SIZE; i++){gk[i+STATE_SIZE_PDDP] = (T) (k == NUM_TIME_STEPS - 1 ? 0.0 : R)*uk[i];}
}
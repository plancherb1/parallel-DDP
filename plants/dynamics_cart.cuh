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
        T *s_xk = &s_x[STATE_SIZE_PDDP*iter];
        T *s_uk = &s_u[NUM_POS*iter];
        T *s_qddk = &s_qdd[NUM_POS*iter];
        T ct = cos(s_xk[1]);     T st = sin(s_xk[1]);            T td2 = s_xk[3]*s_xk[3];
        T H0 = M_CART+M_POLE;    T H1 = MLL_POLE;                T Hod = ML_POLE*ct;
        T TauM = ML_POLE*st;     T Tau0 = TauM * td2 + s_uk[0];  T Tau1 = TauM * GRAVITY;
        // H = [H0 Hod; Hod H1] and Tau = [Tau0; Tau1] and Xdd = H\Tau
        T det = 1/(H0*H1 - Hod*Hod);  
        s_qddk[0] = det * (H1*Tau0 - Hod*Tau1);  s_qddk[1] = det * (H0*Tau1 - Hod*Tau0);
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
    T ct = cos(s_x[1]);              T st = sin(s_x[1]);
    T td = s_x[3];                   T td2 = td*td;
    T H0 = M_CART+M_POLE;            T H1 = MLL_POLE;
    T Hod = ML_POLE*ct;              T TauM = ML_POLE*st;
    T Tau0 = TauM * td2 + s_u[0];    T Tau1 = TauM * GRAVITY;
    // H = [H0 Hod; Hod H1] and Tau = [Tau0; Tau1] and s_xdd = H\Tau
    T det = H0*H1 - Hod*Hod;         T idet = 1/det;
    T xddM = H1*Tau0 - Hod*Tau1;     T thetaddM = H0*Tau1 - Hod*Tau0;
    T thetadd_du = idet * (-Hod);    T thetadd_dthetad = idet * (-2*Hod*TauM*td);
    T xdd_du = idet * (H1);          T xdd_dthetad = idet * (2*H1*TauM*td);
    T Hod_dtheta = -TauM;            T Tau0_dtheta = Hod*td2;      T Tau1_dTheta = Hod*GRAVITY;     
    // Then the complex computes
    T thetaddM_dtheta = H0*Tau1_dTheta - (Hod_dtheta*Tau0 + Hod*Tau0_dtheta);
    T xddM_dtheta = H1*Tau0_dtheta - (Hod_dtheta*Tau1 + Hod*Tau1_dTheta);
    T idet_dtheta = -2*Hod*TauM*idet*idet;
    T thetadd_dtheta = idet * thetaddM_dtheta + idet_dtheta * thetaddM;
    T xdd_dtheta = idet * xddM_dtheta + idet_dtheta * xddM;
    // Then form dqdd
    s_dqdd[0] = 0;             s_dqdd[1] = 0;                // xdd_dx, thetadd_dx
    s_dqdd[2] = xdd_dtheta;    s_dqdd[3] = thetadd_dtheta;
    s_dqdd[4] = 0;             s_dqdd[5] = 0;                // xdd_dxd, thetadd_dxd
    s_dqdd[6] = xdd_dthetad;   s_dqdd[7] = thetadd_dthetad;
    s_dqdd[8] = xdd_du;        s_dqdd[9] = thetadd_du;
}
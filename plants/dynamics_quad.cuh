/*****************************************************************
 * Quad Dynamics
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
#define MASS 0.5
#define invMASS 2
#define LENGTH 0.1750
#define I_0 0.0023 //xx
#define I_1 0
#define I_2 0
#define I_3 0
#define I_4 0.0023 //yy
#define I_5 0
#define I_6 0
#define I_7 0
#define I_8 0.004 //zz
#define Ixx I_0
#define Iyy I_4
#define Izz I_8
#define invIxx 434.782608696
#define invIyy 434.782608696
#define invIzz 250

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
        // States[x_,y_,z_,r,p,y,xd_,yd_,zd_,rd,pd,yd]
        T cX3 = cos(s_xk[3]);   T cX4 = cos(s_xk[4]);   T cX5 = cos(s_xk[5]);
        T sX3 = sin(s_xk[3]);   T sX4 = sin(s_xk[4]);   T sX5 = sin(s_xk[5]);
        T X910 = s_xk[9]*s_xk[10];       T X911 = s_xk[9]*s_xk[11];
        T X1011 = s_xk[10]*s_xk[11];     T X11_2 = s_xk[11]*s_xk[11];

        T sumU = s_uk[0] + s_uk[1] + s_uk[2] + s_uk[3];
        s_qddk[0] = invMASS*sumU*(sX3*sX5 + cX3*cX5*sX4);
        s_qddk[1] = -invMASS*sumU*(cX5*sX3 - cX3*sX4*sX5);
        s_qddk[2] = GRAVITY + invMASS*sumU*cX3*cX4;

        T diffU4 = s_uk[0]-s_uk[1]+s_uk[2]-s_uk[3];     T diffU2 = s_uk[2]-s_uk[0];
        T invcX4 = 1/cX4;    T cX3_2 = cX3*cX3;   T cX34 = cX3*cX4;    T s2X3 = 2.0*sX3*cX3;   T c2X3 = cos(2.0*s_x[3]);
        s_qddk[3] = invcX4*(0.0005434782609*(32000.0*X1011 + 140000.0*(s_u[1]-s_u[3])*cX4 - 28320.0*X910*sX4 - 30160.0*X1011*cX3_2 + 1127.0*diffU4*cX3*sX4 - 140000.0*diffU2*sX3*sX4 + 30160.0*X910*cX3_2*sX4 - 30160.0*X1011*cX34*cX34 + 30160.0*s_x[10]*s_x[10]*cX3*cX4*sX3 - 30160.0*X11_2*cX3*cX4*sX3 + 30160.0*X911*cX3*cX4*sX3*sX4)); 
        s_qddk[4] = 76.08695652*diffU2*cX3 - 0.6125*diffU4*sX3 - 1.0*X911*cX4 - 8.195652174*X910*s2X3 - 16.39130435*X11_2*cX3_2*cX4*sX4 + 16.39130435*X911*cX3_2*cX4 + 16.39130435*X1011*cX3*sX3*sX4;
        s_qddk[5] = -invcX4*(0.0005434782609*(13240.0*X910 - 1127.0*diffU4*cX3 - 140000.0*diffU2*sX3 - 16920.0*X1011*sX4 + 7540.0*X11_2*s2X3*2.0*sX4*cX4 - 15080.0*X910*c2X3 - 15080.0*X911*s2X3*cX4 + 15080.0*X1011*c2X3*sX4));
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
    // zero out s_dqdd
    memset(s_dqdd,0,NUM_POS*(STATE_SIZE+CONTROL_SIZE)*sizeof(T));

    // precompute sin and cos
    T sX3 = sin(s_x[3]); T sX4 = sin(s_x[4]); T sX5 = sin(s_x[5]);
    T cX3 = cos(s_x[3]); T cX4 = cos(s_x[4]); T cX5 = cos(s_x[5]);

    // Rows 0-3 (xdd, ydd, zdd) are pretty straight forward 
    T UMTerm = (s_u[0] + s_u[1] + s_u[2] + s_u[3])*invMASS;
    T row6Term = (sX3*sX5 + cX3*cX5*sX4)*invMASS;
    s_dqdd[0+3*NUM_POS] = (cX3*sX5 - cX5*sX3*sX4)*UMTerm;
    s_dqdd[0+4*NUM_POS] = (cX3*cX4*cX5)*UMTerm;
    s_dqdd[0+5*NUM_POS] = (cX5*sX3 - cX3*sX4*sX5)*UMTerm;
    s_dqdd[0+12*NUM_POS] = row6Term;
    s_dqdd[0+13*NUM_POS] = row6Term;
    s_dqdd[0+14*NUM_POS] = row6Term;
    s_dqdd[0+15*NUM_POS] = row6Term;

    T row7Term = -(cX5*sX3 - cX3*sX4*sX5)*invMASS;
    s_dqdd[1+3*NUM_POS] = -(cX3*cX5 + sX3*sX4*sX5)*UMTerm;
    s_dqdd[1+4*NUM_POS] = (cX3*cX4*sX5)*UMTerm;
    s_dqdd[1+5*NUM_POS] = (sX3*sX5 + cX3*cX5*sX4)*UMTerm;
    s_dqdd[1+12*NUM_POS] = row7Term;
    s_dqdd[1+13*NUM_POS] = row7Term;
    s_dqdd[1+14*NUM_POS] = row7Term;
    s_dqdd[1+15*NUM_POS] = row7Term;

    T row8Term = (cX3*cX4)/MASS;
    s_dqdd[2+3*NUM_POS] = -(cX4*sX3)*UMTerm;
    s_dqdd[2+4*NUM_POS] = -(cX3*sX4)*UMTerm;
    s_dqdd[2+12*NUM_POS] = row8Term;
    s_dqdd[2+13*NUM_POS] = row8Term;
    s_dqdd[2+14*NUM_POS] = row8Term;
    s_dqdd[2+15*NUM_POS] = row8Term;

    // last 3 (rdd, pdd, ydd) get ugly so precompute powers and mults that we need for all those rows
    T X11_2 = s_x[11]*s_x[11];
    T X910 = s_x[9]*s_x[10];
    T X911 = s_x[9]*s_x[11];
    T X1011 = s_x[10]*s_x[11];
    T cX3_2 = cX3*cX3;
    T cX4_2 = cX4*cX4;
    T cX3_2X4 = cX3_2*cX4;
    T cX3_2X4_2 = cX3_2*cX4_2;
    T cX34sX3 = cX3*cX4*sX3;
    T cX3sX34 = cX3*sX3*sX4;
    T cX34sX34 = cX34sX3*sX4;
    T invcX4 = 1.0/cX4;
    T invcX4_2 = invcX4/cX4;
    T sumDifU02 = -s_u[0]+s_u[2];
    T sumDifU0123 = -s_u[0]+s_u[1]-s_u[2]+s_u[3];
    //Inertia Constant: 0.73913043584 = 434.782608696*(I_8-I_0) == (-184.78260896*I_0 - 250.0*I_4 + 434.78260896*I_8) == (-4250/23
    //Length Mass Inertia Constants 2 and 3 = 1127/184 and 14000/184 or 49/8 and 1750/23 == 76.0869565217 & 6.125

    // then terms just for row 3
    T row3Term1 = 6.125*cX3*sX4*invcX4;
    T row3Term2 = 76.0869565217*sX3;
    s_dqdd[3+3*NUM_POS] = invcX4*(76.0869565217*cX3*sX4*sumDifU02 + 6.125*sX3*sX4*sumDifU0123 + 0.73913043584*(s_x[10]*s_x[10]*(2*cX3_2*cX4-cX4) + X11_2*cX4 - 2*X11_2*cX3_2*cX4 - X911*sX4*cX4 + 2.0*X1011*sX3*cX3 + 2*X911*cX3_2*cX4*sX4 - 2*s_x[11]*cX3*cX4_2*sX3 - 2*s_x[10] - 2*X910*cX3*sX3*sX4));
    s_dqdd[3+4*NUM_POS] = invcX4_2*(X910 + X1011*sX4 + 76.0869565217*sX3*sumDifU02 - 6.125*cX3*sumDifU0123 + 0.73913043584*(X910*(1 + cX3_2) + X1011*(1 - cX3_2*sX4 + cX3_2X4_2*sX4) + X911*cX34sX3*cX4_2));
    s_dqdd[3+9*NUM_POS] = invcX4*(s_x[10]*sX4 + 0.73913043584*(s_x[10]*(cX3_2-1)*sX4 + s_x[11]*cX34sX34));
    s_dqdd[3+10*NUM_POS] = invcX4*(s_x[11] + s_x[9]*sX4 - 0.73913043584*(s_x[11]*(1 + cX3_2 + cX3_2X4_2) - s_x[9]*(sX4 + cX3_2*sX4) - 2*s_x[10]*cX34sX3));
    s_dqdd[3+11*NUM_POS] = invcX4*(s_x[10] - 0.73913043584*(s_x[10] + s_x[10]*cX3_2 + s_x[9]*cX34sX34 - 2*s_x[11]*cX34sX3 - s_x[10]*cX3_2X4_2));
    s_dqdd[3+12*NUM_POS] = row3Term1 - row3Term2; // 7/184
    s_dqdd[3+13*NUM_POS] = 76.0869565217 - row3Term1;
    s_dqdd[3+14*NUM_POS] = row3Term1 + row3Term2;
    s_dqdd[3+15*NUM_POS] = -76.0869565217 - row3Term1;

    // then terms just for row 4
    T row4Term1 = 6.125*sX3;
    T row4Term2 = 76.0869565217*cX3;
    s_dqdd[4+3*NUM_POS] = 6.125*cX3*sumDifU0123 - 76.0869565217*sX3*sumDifU02 + 0.73913043584*(X910*(1-2*cX3_2) + X1011*sX4*(2*cX3_2-1) + 2*X11_2*cX34sX34 - 2*X911*cX34sX3);
    s_dqdd[4+4*NUM_POS] = 0.73913043584*(X11_2*cX3_2 - 2.0*X11_2*cX3_2X4_2 - cX3_2*sX4 + X1011*cX34sX3) + X911*sX4;
    s_dqdd[4+9*NUM_POS] = 0.73913043584*(s_x[11]*cX3_2X4 - s_x[10]*sX3*cX3) - s_x[11]*cX4;
    s_dqdd[4+10*NUM_POS] = 0.73913043584*(s_x[11]*cX3sX34 - s_x[9]*sX3*cX3);
    s_dqdd[4+11*NUM_POS] = 0.73913043584*(s_x[9]*cX3_2X4 + s_x[10]*cX3sX34 - 2.0*s_x[11]*cX3_2X4*sX4) - s_x[9]*cX4;
    s_dqdd[4+12*NUM_POS] = -row4Term2 - row4Term1;
    s_dqdd[4+13*NUM_POS] = row4Term1;
    s_dqdd[4+14*NUM_POS] = row4Term2 - row4Term1;
    s_dqdd[4+15*NUM_POS] = row4Term1;

    // then terms just for row 5
    T row5Term1 = -6.125*cX3*invcX4; // 1127/184 and 14000/184 or 49/8 and 1750/23
    T row5Term2 = 76.0869565217*sX3*invcX4; 
    s_dqdd[5+3*NUM_POS] = invcX4*(76.0869565217*cX3*sumDifU02 + 6.125*sX3*sumDifU0123 +  0.73913043584*(X11_2*(sX4*cX4 - 2.0*cX3_2X4*sX4) + X911*(2.0*cX3_2X4 - cX4) - 2.0*X910*sX3*cX3 + 2.0*X1011*cX3sX34));
    s_dqdd[5+4*NUM_POS] = invcX4_2*(X1011 + sX4*(X910 + 76.0869565217*sX3*sumDifU02 - 6.125*cX3*sumDifU0123) - 0.73913043584*(X1011*(1 + cX3_2) + X910*(sX4 - cX3_2*sX4) + X11_2*cX34sX3*cX4_2));
    s_dqdd[5+9*NUM_POS] = invcX4*(s_x[10] + 0.73913043584*(s_x[10]*(cX3_2 - 1) + s_x[11]*cX34sX3));
    s_dqdd[5+10*NUM_POS] = invcX4*(s_x[9] + s_x[11]*sX4  - 0.73913043584*(s_x[11]*(cX3_2*sX4 + 1) - s_x[9]*(1 + cX3_2)));
    s_dqdd[5+11*NUM_POS] = invcX4*(s_x[10]*sX4 + 0.73913043584*(s_x[9]*cX34sX3 - s_x[10]*sX4*(1.0+cX3_2) - 0.25*s_x[11]*cX34sX34));
    s_dqdd[5+12*NUM_POS] = -row5Term1 - row5Term2;
    s_dqdd[5+13*NUM_POS] = row5Term1;
    s_dqdd[5+14*NUM_POS] = -row5Term1 + row5Term2;
    s_dqdd[5+15*NUM_POS] = row5Term1;
}
/*******
nvcc -std=c++11 -o timeDyn.exe timeDyn.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -O3
*******/

#define PLANT 4
#define EE_COST_PDDP 0
#define ERR_TOL 0.1
#include "../config.cuh"
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <stack>
#define NUM_TESTS (1000*100)
#define RANDOM_MEAN 0
#define RANDOM_STDEVq 2
#define RANDOM_STDEVqd 5
#define RANDOM_STDEVu 50
std::default_random_engine randEng(time(0)); //seed
std::normal_distribution<double> randDistq(RANDOM_MEAN, RANDOM_STDEVq); //mean followed by stdiv
std::normal_distribution<double> randDistqd(RANDOM_MEAN, RANDOM_STDEVqd); //mean followed by stdiv
std::normal_distribution<double> randDistu(RANDOM_MEAN, RANDOM_STDEVu); //mean followed by stdiv

//
// Timer that uses clock_gettime()
//
// Based on timer.hpp from Pinocchio
//

/* Return the time spent in secs. */
inline double operator-(const struct timespec & t1,const struct timespec & t0)
{
   /* TODO: double check the double conversion from long (on 64x). */
   return double(t1.tv_sec - t0.tv_sec)+1e-9*double(t1.tv_nsec - t0.tv_nsec);
}

struct TicToc
{
    enum Unit { S = 1, MS = 1000, US = 1000000, NS = 1000000000 };
    Unit DEFAULT_UNIT;
    static std::string unitName(Unit u)
    {
        switch(u) { case S: return "s"; case MS: return "ms"; case US: return "us"; case NS: return "ns"; }
        return "";
    }

    std::stack<struct timespec> stack;
    mutable struct timespec t0;

    TicToc( Unit def = MS ) : DEFAULT_UNIT(def) {}

    inline void tic() {
        stack.push(t0);
        clock_gettime(CLOCK_MONOTONIC,&(stack.top()));
    }

    inline double toc(const Unit factor)
    {
        clock_gettime(CLOCK_MONOTONIC,&t0);
        double dt = (t0-stack.top())*factor;
        stack.pop();
        return dt;
    }
    inline void toc(std::ostream & os, double SMOOTH=1)
    {
        os << toc(DEFAULT_UNIT)/SMOOTH << " " << unitName(DEFAULT_UNIT) << std::endl;
    }
};


template <typename T>
__global__
void forwardDynKern(T *x, T *u, T *qdd, T *I, T *Tbody, int ld_x, int ld_u, int ld_qdd){
    T *xk = x;      T *uk = u;      T *qddk = qdd;
    #pragma unroll
    for (int i=0; i<NUM_TESTS; i++){
        dynamics<T>(qddk,xk,uk,I,Tbody);
        qddk += ld_qdd;     xk += ld_x;     uk += ld_u;
    }    
}

template <typename T>
__host__
void forwardDyn(T *x, T *u, T *qdd, T *I, T *Tbody, int ld_x, int ld_u, int ld_qdd){
    T *xk = x;      T *uk = u;      T *qddk = qdd;
    #pragma unroll
    for (int i=0; i<NUM_TESTS; i++){
        dynamics<T>(qddk,xk,uk,I,Tbody);
        qddk += ld_qdd;     xk += ld_x;     uk += ld_u;
    }    
}

template <typename T>
__host__
void runTests(){
    // allocate variables
    T *h_x, *h_u, *h_qdd, *h_qdd2, *d_x, *d_u, *d_qdd;
    int ld_x = DIM_x_r;     int ld_u = DIM_u_r;     int ld_qdd = NUM_POS;
    gpuErrchk(cudaMalloc((void**)&d_x,ld_x*NUM_TESTS*sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_u,ld_u*NUM_TESTS*sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_qdd,ld_qdd*NUM_TESTS*sizeof(T)));
    h_x = (T *)malloc(ld_x*NUM_TESTS*sizeof(T));
    h_u = (T *)malloc(ld_u*NUM_TESTS*sizeof(T));
    h_qdd = (T *)malloc(ld_qdd*NUM_TESTS*sizeof(T));
    h_qdd2 = (T *)malloc(ld_qdd*NUM_TESTS*sizeof(T));
    
    // load random states and controls
    for (int k=0; k < NUM_TESTS; k++){
        T *xk = &h_x[ld_x*k];   T *uk = &h_u[ld_u*k];
        for (int i=0; i < NUM_POS; i++){
            xk[i] = static_cast<T>(randDistq(randEng));
            xk[i+NUM_POS] = static_cast<T>(randDistqd(randEng));
            uk[i] = static_cast<T>(randDistu(randEng)); 
        }
    }
    gpuErrchk(cudaMemcpy(d_x,h_x,ld_x*NUM_TESTS*sizeof(T),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u,h_u,ld_u*NUM_TESTS*sizeof(T),cudaMemcpyHostToDevice));

    // allocate and load I and Tbody
    T *h_I, *h_Tbody, *d_I, *d_Tbody;
    h_I = (T *)malloc(36*NUM_POS*sizeof(T));    
    h_Tbody = (T *)malloc(36*NUM_POS*sizeof(T));    
    gpuErrchk(cudaMalloc((void**)&d_I,36*NUM_POS*sizeof(T)));   
    gpuErrchk(cudaMalloc((void**)&d_Tbody,36*NUM_POS*sizeof(T)));
    initI<T>(h_I);  initT<T>(h_Tbody);
    gpuErrchk(cudaMemcpy(d_I, h_I, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Tbody, h_Tbody, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));

    // Set up timer
    TicToc timer(TicToc::NS);   double time_alg;

    // compute grads
    timer.tic();
    forwardDyn<T>(h_x,h_u,h_qdd,h_I,h_Tbody,ld_x,ld_u,ld_qdd);
    time_alg = timer.toc(TicToc::NS)/NUM_TESTS;
    std::cout << "CPU  = " << time_alg << " " << timer.unitName(TicToc::NS) << std::endl;

    dim3 fdGrid(1,1);   dim3 fdThreads(8,7);
    timer.tic();
    forwardDynKern<T><<<fdGrid,fdThreads>>>(d_x,d_u,d_qdd,d_I,d_Tbody,ld_x,ld_u,ld_qdd);
    gpuErrchk(cudaDeviceSynchronize());
    time_alg = timer.toc(TicToc::NS)/NUM_TESTS;
    std::cout << "GPU  = " << time_alg << " " << timer.unitName(TicToc::NS) << std::endl;

    // compare
    gpuErrchk(cudaMemcpy(h_qdd2,d_qdd,ld_qdd*NUM_TESTS*sizeof(T),cudaMemcpyDeviceToHost));
    for (int k=0; k < NUM_TESTS; k++){
        T *qddk = &h_qdd[ld_qdd*k];     T *qddk2 = &h_qdd2[ld_qdd*k];
        for (int i=0; i<NUM_POS; i++){
            T err;
            if (qddk[i] == 0 && qddk2[i] == 0){err = 0;}
            else {err = abs(qddk[i] - qddk2[i]) / (qddk[i] > 0 ? qddk[i] : qddk2[i]);}
            if(err > ERR_TOL){printf("ErrPer[%f] for k[%d]i[%d] of host[%f] vs device[%f]\n",err,k,i,qddk[i],qddk2[i]);}
        }
    }

    // free memory
    gpuErrchk(cudaFree(d_x));   gpuErrchk(cudaFree(d_u));   gpuErrchk(cudaFree(d_qdd));
    free(h_x);  free(h_u);  free(h_qdd);    free(h_qdd2);
}

int main(void) {
    runTests<algType>();
    return 0;
}

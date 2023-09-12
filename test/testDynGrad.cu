/*******
nvcc -std=c++11 -o testDynGrad.exe testDynGrad.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -O3
*******/

#define PLANT 4
#define EE_COST_PDDP 0
#define MPC_MODE 1
#define FINITE_DIFF_EPSILON 0.001
#include "../config.cuh"
#include <random>
#define ERR_TOL 0.1 // percent error
#define RANDOM_MEAN 0
#define RANDOM_STDEVq 2
#define RANDOM_STDEVqd 5
#define RANDOM_STDEVu 50
std::default_random_engine randEng(time(0)); //seed
std::normal_distribution<double> randDistq(RANDOM_MEAN, RANDOM_STDEVq); //mean followed by stdiv
std::normal_distribution<double> randDistqd(RANDOM_MEAN, RANDOM_STDEVqd); //mean followed by stdiv
std::normal_distribution<double> randDistu(RANDOM_MEAN, RANDOM_STDEVu); //mean followed by stdiv

template <typename T>
__global__
void integratorGradientKernFiniteDiff(T *d_AB, T *d_x, T *d_u, T *d_I, T *d_Tbody, int ld_x, int ld_u, int ld_AB){
	__shared__ T s_x[2*STATE_SIZE_PDDP];
	__shared__ T s_u[2*CONTROL_SIZE];
	__shared__ T s_qdd[2*NUM_POS];
	for (int timestep = blockIdx.x; timestep < NUM_TIME_STEPS-1; timestep += gridDim.x){
		for (int outputCol = blockIdx.y; outputCol < STATE_SIZE_PDDP + CONTROL_SIZE; outputCol += gridDim.y){
			T *xk = &d_x[timestep*ld_x];
			T *uk = &d_u[timestep*ld_u];
			T *ABk = &d_AB[timestep*ld_AB*DIM_AB_c + ld_AB*outputCol];
			finiteDiffInner<T>(ABk,xk,uk,s_x,s_u,s_qdd,d_I,d_Tbody,outputCol);
		}
	}
}

template <typename T>
__host__
void integratorGradientFiniteDiff(T *h_AB, T *h_x, T *h_u, T *h_I, T *h_Tbody, int ld_x, int ld_u, int ld_AB){
	T s_x[2*STATE_SIZE_PDDP];
	T s_u[2*CONTROL_SIZE];
	T s_qdd[2*NUM_POS];
	for (int timestep = 0; timestep < NUM_TIME_STEPS-1; timestep++){
		for (int outputCol = 0; outputCol < STATE_SIZE_PDDP + CONTROL_SIZE; outputCol++){
			T *xk = &h_x[timestep*ld_x];
			T *uk = &h_u[timestep*ld_u];
			T *ABk = &h_AB[timestep*ld_AB*DIM_AB_c + ld_AB*outputCol];
			finiteDiffInner<T>(ABk,xk,uk,s_x,s_u,s_qdd,h_I,h_Tbody,outputCol);
		}
	}
}

template <typename T>
__global__
void integratorGradientKernAnalytic(T *d_AB, T *d_x, T *d_u, T *d_I, T *d_Tbody, int ld_x, int ld_u, int ld_AB){
	__shared__ T s_x[STATE_SIZE_PDDP];
	__shared__ T s_u[CONTROL_SIZE];
	__shared__ T s_qdd[NUM_POS];
	__shared__ T s_dqdd[3*NUM_POS*NUM_POS];
	T *xk = &d_x[blockIdx.x*ld_x];
	T *uk = &d_u[blockIdx.x*ld_u];
	T *ABk = &d_AB[blockIdx.x*ld_AB*DIM_AB_c];
	// load in the state and control
	#pragma unroll
	for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < STATE_SIZE_PDDP; ind += blockDim.x*blockDim.y){
		s_x[ind] = xk[ind];      if (ind < CONTROL_SIZE){s_u[ind] = uk[ind];}
	}
	__syncthreads();
	// then compute the dynamics gradient
	_integratorGradient(ABk, s_x, s_u, s_qdd, s_dqdd, d_I, d_Tbody, (T)TIME_STEP, ld_AB);
}

template <typename T>
__host__
void integratorGradientAnalytic(T *h_AB, T *h_x, T *h_u, T *h_I, T *h_Tbody, int ld_x, int ld_u, int ld_AB){
	T s_qdd[NUM_POS];
	T s_dqdd[3*NUM_POS*NUM_POS];
	for (int k = 0; k < NUM_TIME_STEPS-1; k++){
		T *xk = &h_x[k*ld_x];
		T *uk = &h_u[k*ld_u];
		T *ABk = &h_AB[k*ld_AB*DIM_AB_c];
		_integratorGradient(ABk, xk, uk, s_qdd, s_dqdd, h_I, h_Tbody, (T)TIME_STEP, ld_AB);
	}
}

// mode CPU = 0, GPU = 1
template <typename T, bool MODE = 0>
__host__
void testDynGrad(){
	// allocate
	T *h_x, *h_u, *h_AB, *h_AB2;
	int ld_AB = DIM_AB_r;	int ld_x = DIM_x_r;		int ld_u = DIM_u_r;
	h_x = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	h_u = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));
	h_AB = (T *)malloc(ld_AB*DIM_AB_c*NUM_TIME_STEPS*sizeof(T));
	h_AB2 = (T *)malloc(ld_AB*DIM_AB_c*NUM_TIME_STEPS*sizeof(T));
	#if MODE
		T *d_x, *d_u, *d_AB, *d_AB2;
		gpuErrchk(cudaMalloc((void**)&d_x,ld_x*NUM_TIME_STEPS*sizeof(T)));
		gpuErrchk(cudaMalloc((void**)&d_u,ld_u*NUM_TIME_STEPS*sizeof(T)));
		gpuErrchk(cudaMalloc((void**)&d_AB,ld_AB*DIM_AB_c*NUM_TIME_STEPS*sizeof(T)));
		gpuErrchk(cudaMalloc((void**)&d_AB2,ld_AB*DIM_AB_c*NUM_TIME_STEPS*sizeof(T)));
	#endif

	// load random states and controls
	for (int k=0; k < NUM_TIME_STEPS; k++){
		T *xk = &h_x[ld_x*k];	T *uk = &h_u[ld_u*k];
		for (int i=0; i < NUM_POS; i++){
			xk[i] = static_cast<T>(randDistq(randEng));
			xk[i+NUM_POS] = static_cast<T>(randDistqd(randEng));
			uk[i] = static_cast<T>(randDistu(randEng));	
		}
	}
	#if MODE
		gpuErrchk(cudaMemcpy(d_x,h_x,ld_x*NUM_TIME_STEPS*sizeof(T),cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_u,h_u,ld_u*NUM_TIME_STEPS*sizeof(T),cudaMemcpyHostToDevice));
	#endif

	// allocate and load I and Tbody
	T *h_I, *h_Tbody;
	h_I = (T *)malloc(36*NUM_POS*sizeof(T));	
	h_Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	
	initI<T>(h_I);	initT<T>(h_Tbody);
	#if MODE
		T *d_I, *d_Tbody;
		gpuErrchk(cudaMalloc((void**)&d_I,36*NUM_POS*sizeof(T)));	
		gpuErrchk(cudaMalloc((void**)&d_Tbody,36*NUM_POS*sizeof(T)));
		gpuErrchk(cudaMemcpy(d_I, h_I, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_Tbody, h_Tbody, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));
	#endif

	// compute grads
	#if MODE
		dim3 aGrid(NUM_TIME_STEPS,1);
		dim3 fdGrid(1,1);//NUM_TIME_STEPS,STATE_SIZE_PDDP+CONTROL_SIZE);
		dim3 aThreads(8,7);
		dim3 fdThreads(8,7);
		integratorGradientKernAnalytic<T><<<aGrid,aThreads>>>(d_AB,d_x,d_u,d_I,d_Tbody,ld_x,ld_u,ld_AB);
		integratorGradientKernFiniteDiff<T><<<fdGrid,fdThreads>>>(d_AB2,d_x,d_u,d_I,d_Tbody,ld_x,ld_u,ld_AB);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(h_AB,d_AB,ld_AB*DIM_AB_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_AB2,d_AB2,ld_AB*DIM_AB_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToHost));
	#else
		integratorGradientAnalytic<T>(h_AB,h_x,h_u,h_I,h_Tbody,ld_x,ld_u,ld_AB);
		integratorGradientFiniteDiff<T>(h_AB2,h_x,h_u,h_I,h_Tbody,ld_x,ld_u,ld_AB);
	#endif

	// compare and print
	for (int k=0; k < NUM_TIME_STEPS-1; k++){
		T *ABk = &h_AB[ld_AB*DIM_AB_c*k];
		T *AB2k = &h_AB2[ld_AB*DIM_AB_c*k];
		for (int c=0; c < DIM_AB_c; c++){
			for (int r=0; r < DIM_AB_r; r++){
				int ind = c*ld_AB + r;
				T err = abs((ABk[ind] - AB2k[ind])/(ABk[ind] != 0 ? ABk[ind] : (AB2k[ind] != 0 ? AB2k[ind] : 1)));
				if (err > ERR_TOL){
					printf("k[%d] c,r[%d,%d]=ind[%d] has err[%.8f] for analytical[%.8f] vs finiteDiff[%.8f]\n",k,c,r,ind,err,ABk[ind],AB2k[ind]);
				}
			}
		}
	}

	//free
	free(h_x);		free(h_u);		free(h_AB);		free(h_I);		free(h_Tbody);
	#if MODE
		cudaFree(d_x);	cudaFree(d_u);	cudaFree(d_AB);	cudaFree(d_I);	cudaFree(d_Tbody);
	#endif
}

char errMsg[]  = "Error: Unkown code - usage is [C]PU or [G]PU\n";
int main(int argc, char *argv[])
{
	srand(time(NULL));
	char hardware = '?'; // require user input
	if (argc > 1){hardware = argv[1][0];}
	if (hardware == 'C'){testDynGrad<algType,0>();}
	else if (hardware == 'G'){testDynGrad<algType,1>();}
	else{printf("%s",errMsg); hardware = '?';}
	return 0;
}
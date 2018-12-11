/*****************************************************************
 * DDP Next Iteration Setup and Init Helper Functions
 * (currently only supports iLQR - UDP in future release)
 *
 *  nextIterationSetup(C/G)PU
 *    costGradientHessian(Threaded/Kern)
 *	  integratorGradient(Threaded/Kern)
 *	  memcpyCurrAKern -- GPU only
 *  initAlg(C/G)PU
 *    (forwardSim(Kern)) (from fpHelpers)	
 *    costGradientHessian(Threaded/Kern)
 *	  integratorGradient(Threaded/Kern)
 *	  memcpyCurrAKern -- GPU only
 *	acceptRejectTraj(C/G)PU
 *	loadVars(C/G)PU
 *  storeVars(C/G)PU
 *	  defect(Comp/Kern)
 *  allocateMemory(C/G)PU
 *  freeMemory(C/G)PU
 *****************************************************************/

template <typename T>
__global__
void memcpyCurrAKern(T **d_A, T *curr_A, int alphaIndex, int size_A){
	int alpha = blockIdx.x;
	int blockSize = blockDim.x;
	if (alpha >= alphaIndex){alpha++;}
 	int k = threadIdx.x;
 	for(int i=0; i<size_A; i++){
    	d_A[alpha][blockSize*i + k] = curr_A[blockSize*i + k];
 	}
}

template <typename T>
__host__ __device__
void memcpyArr(T **dsts, T *src, size_t size, int amount, int skip = -1){
	for (int i = 0; i < amount; i++){
		if (i == skip){continue;}
		memcpy(dsts[i], src, size);
	}
}

// J = (qh-hdes)'Qhand(qh-hdes) + uRu + xQxHandx
template <typename T>
__global__
void costGradientHessianKern(T *d_x, T *d_u, T *d_g, T *d_H, T *d_xg, int ld_x, int ld_u, int ld_H, int ld_g, T *d_Tbody = nullptr, T *d_JT = nullptr){
	#if EE_COST
		__shared__ T s_x[STATE_SIZE];	__shared__ T s_u[NUM_POS];
		__shared__ T s_sinq[NUM_POS];	__shared__ T s_cosq[NUM_POS];
		__shared__ T s_Tb[36*NUM_POS]; 	__shared__ T s_dTb[36*NUM_POS];
		__shared__ T s_T[36*NUM_POS]; 	__shared__ T s_dT[36*NUM_POS*NUM_POS];
		__shared__ T s_eePos[6];		__shared__ T s_deePos[6*NUM_POS];
		for (int k = blockIdx.x; k < NUM_TIME_STEPS; k += gridDim.x){
			T *xk = &d_x[k*ld_x];			T *uk = &d_u[k*ld_u];
			T *Hk = &d_H[k*ld_H*DIM_H_c];	T *gk = &d_g[k*ld_g];
			// load in the states and controls
			#pragma unroll
			for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < STATE_SIZE; ind += blockDim.x*blockDim.y){
				s_x[ind] = xk[ind];		if (ind < NUM_POS){s_u[ind] = uk[ind];}
			}
			__syncthreads();
			// then compute the end effector position and gradient
			compute_eePos<T>(s_T,s_eePos,s_dT,s_deePos,s_sinq,s_Tb,s_dTb,s_x,s_cosq,d_Tbody);
			__syncthreads();
			// then compute the cost grad
			costGrad<T>(Hk,gk,s_eePos,s_deePos,d_xg,s_x,s_u,k,ld_H,d_JT);
		}
	#else
		#pragma unroll
		for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < NUM_TIME_STEPS; k += blockDim.x*gridDim.x){
			T *xk = &d_x[k*ld_x];			T *uk = &d_u[k*ld_u];
			T *Hk = &d_H[k*ld_H*DIM_H_c];	T *gk = &d_g[k*ld_g];
	    	costGrad(Hk,gk,xk,uk,d_xg,k,ld_H);
	 	}
	#endif
}

template <typename T>
__host__
void costGradientHessianThreaded(threadDesc_t desc, T *x, T *u, T *g, T *H, T *xg, int ld_x, int ld_u, int ld_H, int ld_g, T *Tbody = nullptr, T *JT = nullptr){
	#if EE_COST // need lots of temp mem space for the xfrm mats
		T s_sinq[NUM_POS];		T s_cosq[NUM_POS];
		T s_Tb[36*NUM_POS];		T s_dTb[36*NUM_POS];
		T s_T[36*NUM_POS];		T s_dT[36*NUM_POS*NUM_POS];
		T s_eePos[6];			T s_deePos[6*NUM_POS];
		// if JT passed in then first zero the cost
       	if (JT != nullptr){JT[desc.tid] = 0.0;}
	#endif
   	// for each rep on this thread
   	for (unsigned int rep=0; rep<desc.reps; rep++){
      	int kInd = (desc.tid+rep*desc.dim);
      	T *xk = &x[kInd*ld_x];			T *uk = &u[kInd*ld_u];
      	T *Hk = &H[kInd*ld_H*DIM_H_c];	T *gk = &g[kInd*ld_g];
		#if EE_COST 
			// compute the end effector position and gradient
			compute_eePos<T>(s_T,s_eePos,s_dT,s_deePos,s_sinq,s_Tb,s_dTb,xk,s_cosq,Tbody);
			// then compute the cost gradient
			costGrad<T>(Hk,gk,s_eePos,s_deePos,xg,xk,uk,kInd,ld_H,JT,desc.tid);
		#else // simple
			costGrad(Hk,gk,xk,uk,xg,kInd,ld_H);
    	#endif
	}
}

#if USE_FINITE_DIFF
	template <typename T>
	__host__ __device__ __forceinline__
	void finiteDiffInner(T *ABk, T *xk, T *uk, T *s_x, T *s_u, T *s_qdd, T *d_I, T *d_Tbody, int outputCol){
		int start, delta; singleLoopVals(&start,&delta);
		#pragma unroll
		for (int ind = start; ind < STATE_SIZE; ind += delta){
			T val = xk[ind];
			T adj = (outputCol == ind ? FINITE_DIFF_EPSILON : 0.0);
			s_x[ind] = val + adj;
			s_x[ind + STATE_SIZE] = val - adj;
			if (ind < CONTROL_SIZE){
				val = uk[ind];
				adj = (outputCol == ind + STATE_SIZE ? FINITE_DIFF_EPSILON : 0.0);
				s_u[ind] = val + adj;
				s_u[ind + CONTROL_SIZE] = val - adj;
			}
		}
		hd__syncthreads();
		// run dynamics on both states
		dynamics<T>(s_qdd,s_x,s_u,d_I,d_Tbody,nullptr,2);
		hd__syncthreads();
		// now do the finite diff rule
		#pragma unroll
		for (int ind = start; ind < STATE_SIZE; ind += delta){
			T delta = ind < NUM_POS ? (s_x[ind + NUM_POS] - s_x[ind + NUM_POS + STATE_SIZE]) : (s_qdd[ind-NUM_POS] - s_qdd[ind]);
			T dxdd = delta / (2.0*FINITE_DIFF_EPSILON);
			ABk[ind] = (ind == outputCol ? 1.0 : 0.0) + TIME_STEP * dxdd;
		}
	}

	template <typename T>
	__global__
	void integratorGradientKern(T *d_AB, T *d_x, T *d_u, T *d_I, T *d_Tbody, int ld_x, int ld_u, int ld_AB){
		__shared__ T s_x[2*STATE_SIZE];
		__shared__ T s_u[2*CONTROL_SIZE];
		__shared__ T s_qdd[2*NUM_POS];
		for (int timestep = blockIdx.x; timestep < NUM_TIME_STEPS-1; timestep += gridDim.x){
			for (int outputCol = blockIdx.y; outputCol < STATE_SIZE + CONTROL_SIZE; outputCol += gridDim.y){
				T *xk = &d_x[timestep*ld_x];
				T *uk = &d_u[timestep*ld_u];
				T *ABk = &d_AB[timestep*ld_AB*DIM_AB_c + ld_AB*outputCol];
				finiteDiffInner<T>(ABk,xk,uk,s_x,s_u,s_qdd,d_I,d_Tbody,outputCol);
			}
		}
	}

	template <typename T>
	__host__
	void integratorGradientThreaded(threadDesc_t desc, T *d_x, T *d_u, T *d_AB, int ld_x, int ld_u, int ld_AB, T *I, T *Tbody){
		T s_x[2*STATE_SIZE];
		T s_u[2*CONTROL_SIZE];
		T s_qdd[2*NUM_POS];
		// for each rep on this thread
	   	for (int i=0; i<desc.reps; i++){
			int timestep = (desc.tid+i*desc.dim);
			for (int outputCol = 0; outputCol < STATE_SIZE + CONTROL_SIZE; outputCol++){
				T *xk = &d_x[timestep*ld_x];
				T *uk = &d_u[timestep*ld_u];
				T *ABk = &d_AB[timestep*ld_AB*DIM_AB_c + ld_AB*outputCol];
				finiteDiffInner<T>(ABk,xk,uk,s_x,s_u,s_qdd,I,Tbody,outputCol);
			}
		}
	}
# else
	template <typename T>
	__global__
	void integratorGradientKern(T *d_AB, T *d_x, T *d_u, T *d_I, T *d_Tbody, int ld_x, int ld_u, int ld_AB){
		__shared__ T s_x[STATE_SIZE];
		__shared__ T s_u[CONTROL_SIZE];
		__shared__ T s_qdd[NUM_POS];
		__shared__ T s_dqdd[3*NUM_POS*NUM_POS];
		T *xk = &d_x[blockIdx.x*ld_x];
		T *uk = &d_u[blockIdx.x*ld_u];
		T *ABk = &d_AB[blockIdx.x*ld_AB*DIM_AB_c];
		// load in the state and control
		#pragma unroll
		for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < STATE_SIZE; ind += blockDim.x*blockDim.y){
			s_x[ind] = xk[ind];      if (ind < CONTROL_SIZE){s_u[ind] = uk[ind];}
		}
		__syncthreads();
		// then compute the dynamics gradient
		_integratorGradient(ABk, s_x, s_u, s_qdd, s_dqdd, d_I, d_Tbody, (T)TIME_STEP, ld_AB);
	}

	template <typename T>
	__host__
	void integratorGradientThreaded(threadDesc_t desc, T *d_x, T *d_u, T *d_AB, int ld_x, int ld_u, int ld_AB, T *I, T *Tbody){
	   	// for each rep on this thread
	   	for (unsigned int i=0; i<desc.reps; i++){
			int k = (desc.tid+i*desc.dim);
			T *xk = &d_x[k*ld_x*DIM_x_c];
			T *uk = &d_u[k*ld_u*DIM_u_c];
			T *ABk = &d_AB[k*ld_AB*DIM_AB_c]; 
			T s_x[STATE_SIZE];
			T s_u[CONTROL_SIZE];
			T s_qdd[NUM_POS];
			T s_dqdd[3*NUM_POS*NUM_POS];
			#pragma unroll
			for (int ind = 0; ind < STATE_SIZE; ind++){
	        	s_x[ind] = xk[ind];		if (ind < CONTROL_SIZE){s_u[ind] = uk[ind];}
	      	}
			_integratorGradient(ABk, s_x, s_u, s_qdd, s_dqdd, I, Tbody, (T)TIME_STEP, ld_AB);
	   	}
	}
#endif

template <typename T>
__host__ __forceinline__
void nextIterationSetupGPU(T **d_x, T **h_d_x, T *d_xp, T **d_u, T **h_d_u, T *d_up, T **d_d, T **h_d_d, T *d_dp, \
                      	   T *d_AB, T *d_H, T *d_g, T *d_P, T *d_p, T *d_Pp, T *d_pp, T *d_xGoal,\
                      	   int *alphaIndex, cudaStream_t *streams, dim3 dynDimms, dim3 intDimms, \
                      	   int ld_x, int ld_u, int ld_d, int ld_AB, int ld_H, int ld_g, int ld_P, int ld_p, \
                      	   T *d_I = nullptr, T *d_Tbody = nullptr){
	// Compute derivatives for next pass(AB,H,g) and copy u into all us and run each in a separate stream
	// also save x, u, J, P, p into prev variables (interleaving calls for maximum stream potential)
	// these need to finish before the backpass
	integratorGradientKern<T><<<intDimms,dynDimms,0,streams[0]>>>(d_AB,h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_I,d_Tbody,ld_x,ld_u,ld_AB);   
	gpuErrchk(cudaPeekAtLastError());
	if(!EE_COST){costGradientHessianKern<T><<<1,NUM_TIME_STEPS,0,streams[1]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_g,d_H,d_xGoal,ld_x,ld_u,ld_H,ld_g);}
	else{costGradientHessianKern<T><<<NUM_TIME_STEPS,dynDimms,0,streams[1]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_g,d_H,d_xGoal,ld_x,ld_u,ld_H,ld_g,d_Tbody);}
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpyAsync(d_Pp,d_P,ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[2]));
	gpuErrchk(cudaMemcpyAsync(d_pp,d_p,ld_p*DIM_p_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[3]));
	// these need to finish before next fp
	if (NUM_ALPHA > 1){
		memcpyCurrAKern<T><<<NUM_ALPHA-1,NUM_TIME_STEPS,0,streams[4]>>>(d_u,h_d_u[*alphaIndex],*alphaIndex,ld_u*DIM_u_c); gpuErrchk(cudaPeekAtLastError());
		memcpyCurrAKern<T><<<NUM_ALPHA-1,NUM_TIME_STEPS,0,streams[5]>>>(d_x,h_d_x[*alphaIndex],*alphaIndex,ld_x*DIM_x_c); gpuErrchk(cudaPeekAtLastError());
		if (M_F > 1){ memcpyCurrAKern<T><<<NUM_ALPHA-1,NUM_TIME_STEPS,0,streams[6]>>>(d_d,h_d_d[*alphaIndex],*alphaIndex,ld_d*DIM_d_c); gpuErrchk(cudaPeekAtLastError());}
	}
	gpuErrchk(cudaMemcpyAsync(d_xp,h_d_x[*alphaIndex],ld_x*DIM_x_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[7]));
	gpuErrchk(cudaMemcpyAsync(d_up,h_d_u[*alphaIndex],ld_u*DIM_u_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[8]));
	if (M_F > 1){gpuErrchk(cudaMemcpyAsync(d_dp,h_d_d[*alphaIndex],ld_d*DIM_d_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[9]));}
	// synch on backpass streams
	gpuErrchk(cudaStreamSynchronize(streams[0])); gpuErrchk(cudaStreamSynchronize(streams[1])); gpuErrchk(cudaStreamSynchronize(streams[2])); gpuErrchk(cudaStreamSynchronize(streams[3]));
}

template <typename T>
__host__ __forceinline__
void nextIterationSetupCPU(T *x, T *xp, T *u, T *up, T *d, T *dp, T *AB, T *H, T *g, T *P, T *p, T *Pp, T *pp, T *xGoal, \
                           std::thread *threads, int ld_x, int ld_u, int ld_d, int ld_AB, int ld_H, int ld_g, int ld_P, int ld_p, \
                           T *I = nullptr, T *Tbody = nullptr){
    // Compute derivatives for next pass(AB,H,g) also save x, u, J, P, p into prev variables
    threadDesc_t desc;  desc.dim = COST_THREADS;
    for (unsigned int thread_i = 0; thread_i < COST_THREADS; thread_i++){
        desc.tid = thread_i; 	desc.reps = compute_reps(thread_i,COST_THREADS,NUM_TIME_STEPS);
        threads[thread_i] = 
            std::thread(&costGradientHessianThreaded<T>, desc, std::ref(x), std::ref(u), std::ref(g), std::ref(H), std::ref(xGoal), ld_x, ld_u, ld_H, ld_g, std::ref(Tbody), nullptr);
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads,thread_i);}
    }
    desc.dim = INTEGRATOR_THREADS;
    for (unsigned int thread_i = 0; thread_i < INTEGRATOR_THREADS; thread_i++){
        desc.tid = thread_i; 	desc.reps = compute_reps(thread_i,INTEGRATOR_THREADS,(NUM_TIME_STEPS-1));
        threads[COST_THREADS + thread_i] = 
            std::thread(&integratorGradientThreaded<T>, desc, std::ref(x), std::ref(u), std::ref(AB), ld_x, ld_u, ld_AB, std::ref(I), std::ref(Tbody));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, COST_THREADS + thread_i);}
    }
    // also copy P,p into Pp,pp
    threads[COST_THREADS + INTEGRATOR_THREADS] = std::thread(memcpy, std::ref(Pp), std::ref(P), ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
    if(FORCE_CORE_SWITCHES){setCPUForThread(threads, COST_THREADS + INTEGRATOR_THREADS);}
    threads[COST_THREADS + INTEGRATOR_THREADS + 1] = std::thread(memcpy, std::ref(pp), std::ref(p), ld_p*NUM_TIME_STEPS*sizeof(T));
    if(FORCE_CORE_SWITCHES){setCPUForThread(threads, COST_THREADS + INTEGRATOR_THREADS + 1);}
    // these can finish during the backpass before the forward pass (but we sync here because fast and more straightforward)
    threads[COST_THREADS + INTEGRATOR_THREADS + 2] = std::thread(memcpy, std::ref(xp), std::ref(x), ld_x*NUM_TIME_STEPS*sizeof(T));
    if(FORCE_CORE_SWITCHES){setCPUForThread(threads, COST_THREADS + INTEGRATOR_THREADS + 2);}
    threads[COST_THREADS + INTEGRATOR_THREADS + 3] = std::thread(memcpy, std::ref(up), std::ref(u), ld_u*NUM_TIME_STEPS*sizeof(T));
    if(FORCE_CORE_SWITCHES){setCPUForThread(threads, COST_THREADS + INTEGRATOR_THREADS + 3);}
    // synch on threads
    for (unsigned int thread_i = 0; thread_i < COST_THREADS + INTEGRATOR_THREADS + 4; thread_i++){threads[thread_i].join();}
}

template <typename T>
__host__ __forceinline__
void nextIterationSetupCPU2(T **xs, T *xp, T **us, T *up, T **ds, T *dp, T *AB, T *H, T *g, T *P, T *p, T *Pp, T *pp, T *xGoal, \
                           std::thread *threads, int *alphaIndex, int ld_x, int ld_u, int ld_d, int ld_AB, int ld_H, int ld_g, int ld_P, int ld_p, \
                           T *I = nullptr, T *Tbody = nullptr){
	int flag = 1;	if (*alphaIndex == -1){*alphaIndex = 0; flag = 0;}
	nextIterationSetupCPU<T>(xs[*alphaIndex],xp,us[*alphaIndex],up,ds[*alphaIndex],dp,AB,H,g,P,p,Pp,pp,xGoal,threads,ld_x,ld_u,ld_d,ld_AB,ld_H,ld_g,ld_P,ld_p,I,Tbody);
 	// and copy x,u,d to all x,u,d if needed
 	if(flag){
	 	threads[0] = std::thread(&memcpyArr<T>, std::ref(xs), std::ref(xs[*alphaIndex]), ld_x*NUM_TIME_STEPS*sizeof(T), NUM_ALPHA, *alphaIndex);
	 	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
	 	threads[1] = std::thread(&memcpyArr<T>, std::ref(us), std::ref(us[*alphaIndex]), ld_u*NUM_TIME_STEPS*sizeof(T), NUM_ALPHA, *alphaIndex);
	 	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
		if (M_F > 1){
			threads[2] = std::thread(&memcpyArr<T>, std::ref(ds), std::ref(ds[*alphaIndex]), ld_d*NUM_TIME_STEPS*sizeof(T), NUM_ALPHA, *alphaIndex);
			if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2);}
		}
	 	threads[0].join(); threads[1].join(); if (M_F > 1){threads[2].join();}
	}
}

template <typename T>
__host__ __forceinline__
void initAlgGPU(T **d_x, T **h_d_x, T *d_xp, T *d_xp2, T **d_u, T **h_d_u, T *d_up, T **d_d, T **h_d_d, T *d_dp, T *d_dT, T *d_AB, T *d_H, T *d_g, \
             	T *d_KT, T *d_du, T *d_JT, T *prevJ, T *d_xGoal, T *d_alpha, int *alphaIndex, int *alphaOut, T *Jout, \
             	cudaStream_t *streams, dim3 dynDimms, dim3 intDimms, int forwardRolloutFlag, \
             	int ld_x, int ld_u, int ld_d, int ld_AB, int ld_H, int ld_g, int ld_KT, int ld_du, \
             	T *d_I = nullptr, T *d_Tbody = nullptr){
	if (forwardRolloutFlag){alphaOut[0] = 0;}	else{alphaOut[0] = -1;}
	// compute initial derivatives in separate streams and save the current utraj, xtraj, xp, xp2, up, and d_d
	integratorGradientKern<T><<<intDimms,dynDimms,0,streams[0]>>>(d_AB,h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_I,d_Tbody,ld_x,ld_u,ld_AB); gpuErrchk(cudaPeekAtLastError());
	if(!EE_COST){costGradientHessianKern<T><<<1,NUM_TIME_STEPS,0,streams[1]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_g,d_H,d_xGoal,ld_x,ld_u,ld_H,ld_g);}
	else if(!forwardRolloutFlag){costGradientHessianKern<T><<<NUM_TIME_STEPS,dynDimms,0,streams[1]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_g,d_H,d_xGoal,ld_x,ld_u,ld_H,ld_g,d_Tbody,d_JT);}
	else {costGradientHessianKern<T><<<1,dynDimms,0,streams[1]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_g,d_H,d_xGoal,ld_x,ld_u,ld_H,ld_g,d_Tbody);}
	gpuErrchk(cudaPeekAtLastError());
	if (NUM_ALPHA > 1){
		memcpyCurrAKern<<<NUM_ALPHA-1,NUM_TIME_STEPS,0,streams[2]>>>(d_u,h_d_u[*alphaIndex],*alphaIndex,ld_u); gpuErrchk(cudaPeekAtLastError());
		memcpyCurrAKern<<<NUM_ALPHA-1,NUM_TIME_STEPS,0,streams[3]>>>(d_x,h_d_x[*alphaIndex],*alphaIndex,ld_x); gpuErrchk(cudaPeekAtLastError());
		if (M_F > 1){memcpyCurrAKern<T><<<NUM_ALPHA-1,NUM_TIME_STEPS,0,streams[4]>>>(d_d,h_d_d[*alphaIndex],*alphaIndex,ld_d); gpuErrchk(cudaPeekAtLastError());}
	}
	gpuErrchk(cudaMemcpyAsync(d_xp,h_d_x[*alphaIndex],ld_x*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[5]));
	gpuErrchk(cudaMemcpyAsync(d_xp2,h_d_x[*alphaIndex],ld_x*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[6]));
	gpuErrchk(cudaMemcpyAsync(d_up,h_d_u[*alphaIndex],ld_u*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[7]));
	if (M_F > 1){gpuErrchk(cudaMemcpyAsync(d_dp,h_d_d[*alphaIndex],ld_d*DIM_d_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[7]));}

	// get the cost and add epsilon in case the intialization results in zero update for the first pass
	defectKern<T><<<NUM_ALPHA,NUM_TIME_STEPS,0,streams[5]>>>(d_d,d_dT,ld_d); gpuErrchk(cudaPeekAtLastError());
	#if !EE_COST
		costKern<T><<<1,NUM_TIME_STEPS,NUM_TIME_STEPS*sizeof(T),streams[6]>>>(d_x,d_u,d_JT,d_xGoal,ld_x,ld_u);
	#else
		if (forwardRolloutFlag){costKern<T,0><<<1,1,0,streams[6]>>>(d_JT);}
		else{costKern<T,1><<<1,NUM_TIME_STEPS,NUM_TIME_STEPS*sizeof(T),streams[1]>>>(d_JT);}
	#endif
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(prevJ, &d_JT[*alphaIndex], sizeof(T), cudaMemcpyDeviceToHost));
	*prevJ += 2*TOL_COST;
	gpuErrchk(cudaMemcpyAsync(&d_JT[*alphaIndex], prevJ, sizeof(T), cudaMemcpyHostToDevice,streams[7]));
	Jout[0] = *prevJ - 2*TOL_COST;
	gpuErrchk(cudaDeviceSynchronize()); // make sure this all finishes before we start the main loop
}

template <typename T>
__host__ __forceinline__
void initAlgCPU(T *x, T *xp, T *xp2, T *u, T *up, T *AB, T *H, T *g, T *KT, T *du, T *d, T *JT, T *Jout, T *prevJ, T *alpha, int *alphaOut, T *xGoal, \
				std::thread *threads, int forwardRolloutFlag, int ld_x, int ld_u, int ld_AB, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_d, \
				T *I = nullptr, T *Tbody = nullptr){
    threadDesc_t desc;
    if (forwardRolloutFlag){alphaOut[0] = 0;}	else{alphaOut[0] = -1;}
    // compute the cost (if needed) and initial derivatives
    if (!EE_COST){
        desc.dim = COST_THREADS;
        for (unsigned int thread_i = 0; thread_i < COST_THREADS; thread_i++){
            desc.tid = thread_i; 	desc.reps = compute_reps(thread_i,COST_THREADS,NUM_TIME_STEPS);
            threads[thread_i] = std::thread(&costThreaded<T>, desc, std::ref(x), std::ref(u), std::ref(JT), std::ref(xGoal), ld_x, ld_u);
        }
    }
    desc.dim = COST_THREADS;
    for (unsigned int thread_i = 0; thread_i < COST_THREADS; thread_i++){
        desc.tid = thread_i; 	desc.reps = compute_reps(thread_i,COST_THREADS,NUM_TIME_STEPS);
        if (EE_COST && !forwardRolloutFlag){
            threads[COST_THREADS + thread_i] = 
                std::thread(&costGradientHessianThreaded<T>, desc, std::ref(x), std::ref(u), std::ref(g), std::ref(H), std::ref(xGoal), ld_x, ld_u, ld_H, ld_g, std::ref(Tbody), std::ref(JT));    
        }
        else{
            threads[COST_THREADS + thread_i] = 
                std::thread(&costGradientHessianThreaded<T>, desc, std::ref(x), std::ref(u), std::ref(g), std::ref(H), std::ref(xGoal), ld_x, ld_u, ld_H, ld_g, std::ref(Tbody), nullptr);    
        }
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, COST_THREADS + thread_i);}
    }
    desc.dim = INTEGRATOR_THREADS;
    for (unsigned int thread_i = 0; thread_i < INTEGRATOR_THREADS; thread_i++){
        desc.tid = thread_i; 	desc.reps = compute_reps(thread_i,INTEGRATOR_THREADS,(NUM_TIME_STEPS-1));
        threads[2*COST_THREADS + thread_i] = std::thread(&integratorGradientThreaded<T>, desc, std::ref(x), std::ref(u), std::ref(AB), ld_x, ld_u, ld_AB, std::ref(I), std::ref(Tbody));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2*COST_THREADS + thread_i);}
    }  
    // finally copy x,u into xp,xp2,up
    threads[2*COST_THREADS + INTEGRATOR_THREADS] = std::thread(memcpy, std::ref(xp), std::ref(x), ld_x*NUM_TIME_STEPS*sizeof(T));
    if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2*COST_THREADS + INTEGRATOR_THREADS);}
    threads[2*COST_THREADS + INTEGRATOR_THREADS + 1] = std::thread(memcpy, std::ref(xp2), std::ref(x), ld_x*NUM_TIME_STEPS*sizeof(T));
    if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2*COST_THREADS + INTEGRATOR_THREADS + 1);}
    threads[2*COST_THREADS + INTEGRATOR_THREADS + 2] = std::thread(memcpy, std::ref(up), std::ref(u), ld_u*NUM_TIME_STEPS*sizeof(T));
    if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2*COST_THREADS + INTEGRATOR_THREADS + 2);}
    // make sure all threads finish
    for (unsigned int thread_i = COST_THREADS; thread_i < 2*COST_THREADS + INTEGRATOR_THREADS + 3; thread_i++){threads[thread_i].join();}
    // sum the cost and add epsilon in case the intialization results in zero update for the first pass
    *prevJ = 0;     unsigned int cost_threads = EE_COST && forwardRolloutFlag ? FSIM_THREADS : COST_THREADS;
    for (unsigned int thread_i = 0; thread_i < cost_threads; thread_i++){if(!EE_COST){threads[thread_i].join();} *prevJ += JT[thread_i];}
    Jout[0] = *prevJ;   *prevJ *= (1 + 2*TOL_COST); // we are doing % change now so need to account for that
}

template <typename T>
__host__ __forceinline__
void initAlgCPU2(T **xs, T *xp, T *xp2, T **us, T *up, T *AB, T *H, T *g, T *KT, T *du, T **ds, T *JT, T *Jout, T *prevJ, T *alphas, int *alphaOut, T *xGoal, \
				std::thread *threads, int forwardRolloutFlag, int ld_x, int ld_u, int ld_AB, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_d, \
				T *I = nullptr, T *Tbody = nullptr){
	initAlgCPU<T>(xs[0],xp,xp2,us[0],up,AB,H,g,KT,du,ds[0],JT,Jout,prevJ,alphas,alphaOut,xGoal,threads,
		          forwardRolloutFlag,ld_x,ld_u,ld_AB,ld_H,ld_g,ld_KT,ld_du,ld_d,I,Tbody);
	// also copy all x,u,d to xs,us,ds
	for (int i = 1; i < NUM_ALPHA; i++){
		threads[3*i] = std::thread(memcpy, std::ref(xs[i]), std::ref(xs[0]), ld_x*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 3*i);}
        threads[3*i+1] = std::thread(memcpy, std::ref(us[i]), std::ref(us[0]), ld_u*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 3*i+1);}
        if (M_F > 1){
            threads[3*i+2] = std::thread(memcpy, std::ref(ds[i]), std::ref(ds[0]), ld_d*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 3*i+2);}
        }	
	}
	for(int i = 1; i < NUM_ALPHA; i++){
		threads[3*i].join();	threads[3*i+1].join();	if (M_F > 1){threads[3*i+2].join();}
	}
}

template <typename T>
__host__ __forceinline__
int acceptRejectTrajGPU(T **h_d_x, T *d_xp, T **h_d_u, T *d_up, T **h_d_d, T *d_dp, \
                   		T *J, T *prevJ, T *dJ, T *rho, T *drho, int *alphaIndex, int *alphaOut, T *Jout, \
                   		int *iter, cudaStream_t *streams, int ld_x, int ld_u, int ld_d, int max_iter = MAX_ITER, int *updated = nullptr){
	// if failure increase rho, reset x,u,P,p,d
	if (*dJ < 0){
		*drho = max((*drho)*RHO_FACTOR,RHO_FACTOR); *rho = min((*rho)*(*drho), RHO_MAX);
		*alphaIndex = 0; alphaOut[*iter] = -1; Jout[*iter] = *prevJ;
		// gpuErrchk(cudaMemcpyAsync(d_P,d_Pp,ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[0]));
		// gpuErrchk(cudaMemcpyAsync(d_p,d_pp,ld_p*DIM_p_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[1]));
		gpuErrchk(cudaMemcpyAsync(h_d_x[*alphaIndex],d_xp,ld_x*DIM_x_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[0]));
		gpuErrchk(cudaMemcpyAsync(h_d_u[*alphaIndex],d_up,ld_u*DIM_u_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[1]));
		if (M_F > 1){gpuErrchk(cudaMemcpyAsync(h_d_d[*alphaIndex],d_dp,ld_d*DIM_d_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[2]));}
		gpuErrchk(cudaDeviceSynchronize());
		// check for maxRho failure
		if (*rho == RHO_MAX && !IGNORE_MAX_ROX_EXIT){if (DEBUG_SWITCH){printf("Exiting for maxRho\n");} return 1;}
		else if (DEBUG_SWITCH){printf("[!]Forward Pass Failed Increasing Rho\n");}
	}
	// else try to decrease rho if we can and turn dJ into a percentage and save the cost to prevJ for next time and check for cost tol or max iter exit
	else {
		*drho = min((*drho)/RHO_FACTOR, 1.0/RHO_FACTOR); *rho = max((*rho)*(*drho), RHO_MIN);
		*dJ = (*dJ)/(*prevJ); *prevJ = J[*alphaIndex]; alphaOut[*iter] = *alphaIndex; Jout[*iter] = J[*alphaIndex];
		if(updated != nullptr){*updated = 1;}
		// check for convergence
		if(*dJ < TOL_COST){if (DEBUG_SWITCH){printf("Exiting for tolCost[%f]\n",*dJ);} return 1;}      
	}
	// check for max iters
	if (*iter == max_iter){ if (DEBUG_SWITCH){printf("Breaking for MaxIter\n");} return 1;}
	else{*iter += 1;}
	return 0;
}

template <typename T>
__host__ __forceinline__
int acceptRejectTrajCPU(T *x, T *xp, T *u, T *up, T *d, T *dp, T J, T *prevJ, \
						T *dJ, T *rho, T *drho, int *alphaIndex, int *alphaOut, T *Jout, \
                   		int *iter, std::thread *threads, int ld_x, int ld_u, int ld_d, int max_iter = MAX_ITER){
	// if failure increase rho, reset x,u,P,p,d
	if (*alphaIndex == -1){
		*drho = max((*drho)*RHO_FACTOR,RHO_FACTOR); *rho = min((*rho)*(*drho), RHO_MAX);  alphaOut[*iter] = -1; Jout[*iter] = *prevJ;
		// threads[0] = std::thread(memcpy, std::ref(Pp), std::ref(P), ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
     	// threadss[1] = std::thread(memcpy, std::ref(pp), std::ref(p), ld_p*DIM_p_c*NUM_TIME_STEPS*sizeof(T));
     	threads[0] = std::thread(memcpy, std::ref(x), std::ref(xp), ld_x*NUM_TIME_STEPS*sizeof(T));
     	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
     	threads[1] = std::thread(memcpy, std::ref(u), std::ref(up), ld_u*NUM_TIME_STEPS*sizeof(T));
     	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
		if (M_F > 1){
			threads[2] = std::thread(memcpy, std::ref(d), std::ref(dp), ld_d*NUM_TIME_STEPS*sizeof(T));
			if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2);}
		}
     	threads[0].join(); threads[1].join(); if (M_F > 1){threads[2].join();}
		// check for maxRho failure
		if (*rho == RHO_MAX && !IGNORE_MAX_ROX_EXIT){if (DEBUG_SWITCH){printf("Exiting for maxRho\n");} return 1;}
		else if (DEBUG_SWITCH){printf("[!]Forward Pass Failed Increasing Rho\n");}
	}
	// else try to decrease rho if we can and turn dJ into a percentage and save the cost to prevJ for next time and check for cost tol or max iter exit
	else {
		*drho = min((*drho)/RHO_FACTOR, 1.0/RHO_FACTOR); *rho = max((*rho)*(*drho), RHO_MIN);
		*dJ = (*dJ)/(*prevJ); *prevJ = J; alphaOut[*iter] = *alphaIndex; Jout[*iter] = J;
		// check for convergence
		if(*dJ < TOL_COST && 0){if (DEBUG_SWITCH){printf("Exiting for tolCost[%f]\n",*dJ);} return 1;}      
	}
	// check for max iters
	if (*iter == max_iter){ if (DEBUG_SWITCH){printf("Breaking for MaxIter\n");} return 1;}
	else{*iter += 1;}
	return 0;
}

template <typename T>
__host__ __forceinline__
int acceptRejectTrajCPU2(T **xs, T *xp, T **us, T *up, T **ds, T *dp, T J, T *prevJ, \
						T *dJ, T *rho, T *drho, int *alphaIndex, int *alphaOut, T *Jout, \
                   		int *iter, std::thread *threads, int ld_x, int ld_u, int ld_d, int max_iter = MAX_ITER){
	// if failure increase rho, reset x,u,P,p,d
	if (*alphaIndex == -1){
		*drho = max((*drho)*RHO_FACTOR,RHO_FACTOR); *rho = min((*rho)*(*drho), RHO_MAX);  alphaOut[*iter] = -1; Jout[*iter] = *prevJ;
		// threads[0] = std::thread(memcpy, std::ref(Pp), std::ref(P), ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
     	// threadss[1] = std::thread(memcpy, std::ref(pp), std::ref(p), ld_p*DIM_p_c*NUM_TIME_STEPS*sizeof(T));
     	threads[0] = std::thread(&memcpyArr<T>, std::ref(xs), std::ref(xp), ld_x*NUM_TIME_STEPS*sizeof(T), NUM_ALPHA, -1);
     	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
     	threads[1] = std::thread(&memcpyArr<T>, std::ref(us), std::ref(up), ld_u*NUM_TIME_STEPS*sizeof(T), NUM_ALPHA, -1);
     	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
		if (M_F > 1){
			threads[2] = std::thread(&memcpyArr<T>, std::ref(ds), std::ref(dp), ld_d*NUM_TIME_STEPS*sizeof(T), NUM_ALPHA, -1);
			if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2);}
		}
     	threads[0].join(); threads[1].join(); if (M_F > 1){threads[2].join();}
		// check for maxRho failure
		if (*rho == RHO_MAX && !IGNORE_MAX_ROX_EXIT){if (DEBUG_SWITCH){printf("Exiting for maxRho\n");} return 1;}
		else if (DEBUG_SWITCH){printf("[!]Forward Pass Failed Increasing Rho\n");}
	}
	// else try to decrease rho if we can and turn dJ into a percentage and save the cost to prevJ for next time and check for cost tol or max iter exit
	else {
		*drho = min((*drho)/RHO_FACTOR, 1.0/RHO_FACTOR); *rho = max((*rho)*(*drho), RHO_MIN);
		*dJ = (*dJ)/(*prevJ); *prevJ = J; alphaOut[*iter] = *alphaIndex; Jout[*iter] = J;
		// check for convergence
		if(*dJ < TOL_COST && 0){if (DEBUG_SWITCH){printf("Exiting for tolCost[%f]\n",*dJ);} return 1;}      
	}
	// check for max iters
	if (*iter == max_iter){ if (DEBUG_SWITCH){printf("Breaking for MaxIter\n");} return 1;}
	else{*iter += 1;}
	return 0;
}

template <typename T>
__host__ __forceinline__
void loadVarsGPU(T **d_x, T **h_d_x, T *d_xp, T *x0, T **d_u, T **h_d_u, T *d_up, T *u0, T *d_P, T *d_Pp, T *P0, T *d_p, T *d_pp, T *p0, \
			  	 T *d_KT, T *KT0, T *d_du, T *d_dT, T **d_d, T **h_d_d, T *d0, T *d_AB, int *d_err, T *xGoal, T *d_xGoal, T *d_alpha, \
			  	 T *d_Tbody, T *d_I, T *d_JT, int clearVarsFlag, int forwardRolloutFlag, cudaStream_t *streams, dim3 dynDimms, \
			  	 int ld_x, int ld_u, int ld_P, int ld_p, int ld_KT, int ld_du, int ld_d, int ld_AB){
	// load x and u onto the device (assumes passed in x and u are ld aligned -- and goal)
	gpuErrchk(cudaMemcpyAsync(h_d_x[0], x0, ld_x*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[0]));
	gpuErrchk(cudaMemcpyAsync(h_d_u[0], u0, ld_u*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[1]));
	gpuErrchk(cudaMemcpyAsync(d_xp, x0, ld_x*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[2]));
	gpuErrchk(cudaMemcpyAsync(d_up, u0, ld_u*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[3]));
	int goalSize;	if (!EE_COST){goalSize = STATE_SIZE;}	else{goalSize = 6;}
	gpuErrchk(cudaMemcpyAsync(d_xGoal, xGoal, goalSize*sizeof(T), cudaMemcpyHostToDevice, streams[4]));
	// clear vars if requested -- can run in sync with the x0,u0 transfer
  	if (clearVarsFlag){
	    gpuErrchk(cudaMemsetAsync(d_P,0,ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T),streams[4]));
	    gpuErrchk(cudaMemsetAsync(d_Pp,0,ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T),streams[5]));
	    gpuErrchk(cudaMemsetAsync(d_p,0,ld_p*NUM_TIME_STEPS*sizeof(T),streams[6]));
	    gpuErrchk(cudaMemsetAsync(d_pp,0,ld_p*NUM_TIME_STEPS*sizeof(T),streams[7]));
	    gpuErrchk(cudaMemsetAsync(d_KT,0,ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T),streams[8]));
	    for (int i=0; i<NUM_ALPHA; i++){gpuErrchk(cudaMemsetAsync(h_d_d[i],0,ld_d*NUM_TIME_STEPS*sizeof(T),streams[4+i]));}
  	}
  	// load in the vars
  	else{
  		gpuErrchk(cudaMemcpyAsync(d_P,P0,ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[4]));
	    gpuErrchk(cudaMemcpyAsync(d_Pp,P0,ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[5]));
	    gpuErrchk(cudaMemcpyAsync(d_p,p0,ld_p*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[6]));
	    gpuErrchk(cudaMemcpyAsync(d_pp,p0,ld_p*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[7]));
	    gpuErrchk(cudaMemcpyAsync(d_KT,KT0,ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[8]));
	    for (int i=0; i<NUM_ALPHA; i++){gpuErrchk(cudaMemcpyAsync(h_d_d[i],d0,ld_d*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice, streams[4+i]));}
  	}
  	// always clear these
  	gpuErrchk(cudaMemsetAsync(d_du,0,ld_du*NUM_TIME_STEPS*sizeof(T),streams[9]));
    gpuErrchk(cudaMemsetAsync(d_err,0,M_B*sizeof(int),streams[10]));
    gpuErrchk(cudaMemsetAsync(d_dT,0,NUM_ALPHA*sizeof(T),streams[11]));
    T *ABN = d_AB + ld_AB*DIM_AB_c*(NUM_TIME_STEPS-2);
    gpuErrchk(cudaMemsetAsync(ABN,0,ld_AB*DIM_AB_c*sizeof(T),streams[12]));

    // sync on streams needed for rollout
    gpuErrchk(cudaStreamSynchronize(streams[0]));	gpuErrchk(cudaStreamSynchronize(streams[1]));	gpuErrchk(cudaStreamSynchronize(streams[2]));
    gpuErrchk(cudaStreamSynchronize(streams[8]));	gpuErrchk(cudaStreamSynchronize(streams[9]));
    for (int i=0; i<NUM_ALPHA; i++){gpuErrchk(cudaStreamSynchronize(streams[4+i]))};

    // run initial forward sim if asked
	if (forwardRolloutFlag){
		if (!EE_COST){forwardSimKern<T><<<M_F,dynDimms,0,streams[0]>>>(d_x,d_u,d_KT,d_du,d_d,d_alpha,d_xp,ld_x,ld_u,ld_KT,ld_du,ld_d,d_I,d_Tbody);}
		else{forwardSimKern<T><<<M_F,dynDimms,0,streams[0]>>>(d_x,d_u,d_KT,d_du,d_d,d_alpha,d_xp,ld_x,ld_u,ld_KT,ld_du,ld_d,d_I,d_Tbody,d_xGoal,d_JT);}
		gpuErrchk(cudaPeekAtLastError());
	}
   
	// make sure this all completes before we do initial computes
	gpuErrchk(cudaDeviceSynchronize());
}

template <typename T>
__host__ __forceinline__
void loadVarsCPU(T *x, T *xp, T *x0, T *u, T *up, T *u0, T *P, T *Pp, T *P0, T *p, T *pp, T *p0, \
			  	 T *KT, T *KT0, T *du, T *d, T *d0, T *AB, int *err, int clearVarsFlag, int forwardRolloutFlag, \
			  	 T *alpha, T *I, T *Tbody, T *xGoal, T *JT, std::thread *threads, 
			  	 int ld_x, int ld_u, int ld_P, int ld_p, int ld_KT, int ld_du, int ld_AB, int ld_d){
	// load x and u onto the device and into xp, up (assumes passed in x0 and u0 are ld aligned)
	threads[0] = std::thread(memcpy, std::ref(x), std::ref(x0), ld_x*NUM_TIME_STEPS*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
	threads[1] = std::thread(memcpy, std::ref(u), std::ref(u0), ld_u*NUM_TIME_STEPS*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
	threads[2] = std::thread(memcpy, std::ref(xp), std::ref(x0), ld_x*NUM_TIME_STEPS*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2);}
	threads[3] = std::thread(memcpy, std::ref(up), std::ref(u0), ld_u*NUM_TIME_STEPS*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 3);}
	// for K0, P0, p0, d0 either load or clear
	if (clearVarsFlag){
		threads[4] = std::thread(memset, std::ref(P), 0, ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 4);}
		threads[5] = std::thread(memset, std::ref(Pp), 0, ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 5);}
		threads[6] = std::thread(memset, std::ref(p), 0, ld_p*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 6);}
		threads[7] = std::thread(memset, std::ref(pp), 0, ld_p*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 7);}
		threads[8] = std::thread(memset, std::ref(KT), 0, ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 8);}
		threads[9] = std::thread(memset, std::ref(d), 0, ld_d*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 9);}
	} 
	else{
		threads[4] = std::thread(memcpy, std::ref(P), std::ref(P0), ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 4);}
		threads[5] = std::thread(memcpy, std::ref(Pp), std::ref(P0), ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 5);}
		threads[6] = std::thread(memcpy, std::ref(p), std::ref(p0), ld_p*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 6);}
		threads[7] = std::thread(memcpy, std::ref(pp), std::ref(p0), ld_p*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 7);}
		threads[8] = std::thread(memcpy, std::ref(KT), std::ref(KT0), ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 8);}
		threads[9] = std::thread(memcpy, std::ref(d), std::ref(d0), ld_d*NUM_TIME_STEPS*sizeof(T));
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 9);}
	}
	// always clear err and du
	threads[10] = std::thread(memset, std::ref(du), 0, ld_du*NUM_TIME_STEPS*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 10);}
	threads[11] = std::thread(memset, std::ref(err), 0, BP_THREADS*sizeof(int));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 11);}
	T *ABN = AB + ld_AB*DIM_AB_c*(NUM_TIME_STEPS-2);
	threads[12] = std::thread(memset, std::ref(ABN), 0, ld_AB*DIM_AB_c*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 12);}
	// join threads
	threads[0].join();	threads[1].join();	threads[2].join();	threads[3].join();	threads[4].join();
	threads[5].join();	threads[6].join();	threads[7].join();	threads[8].join();	threads[9].join();
	threads[10].join();	threads[11].join();	threads[12].join();

	// rollout if requested
	if (forwardRolloutFlag){
        threadDesc_t desc;	desc.dim = FSIM_THREADS;
        for (unsigned int thread_i = 0; thread_i < FSIM_THREADS; thread_i++){
            desc.tid = thread_i; 	desc.reps = compute_reps(thread_i,FSIM_THREADS,M_F);
            #if EE_COST
                threads[thread_i] = std::thread(&forwardSim<T>, desc, std::ref(x), std::ref(u), std::ref(KT), std::ref(du), std::ref(d), alpha[0], 
                                                        			  std::ref(xp), ld_x, ld_u, ld_KT, ld_du, ld_d, 
                                                        			  std::ref(I), std::ref(Tbody), std::ref(xGoal), std::ref(JT));
            
            #else
                threads[thread_i] = std::thread(&forwardSim<T>, desc, std::ref(x), std::ref(u), std::ref(KT), std::ref(du), std::ref(d), alpha[0], 
                                                        			  std::ref(xp), ld_x, ld_u, ld_KT, ld_du, ld_d, 
                                                        			  std::ref(I), std::ref(Tbody), nullptr, nullptr);
            #endif
            if(FORCE_CORE_SWITCHES){setCPUForThread(threads, thread_i);}
        }
        for (unsigned int thread_i = 0; thread_i < FSIM_THREADS; thread_i++){threads[thread_i].join();}
    }
}

// store vars to CPU and compute total max defect if requested for debug / printing
template <typename T>
__host__ __forceinline__
void storeVarsGPU(T **h_d_x, T *x0, T **h_d_u, T *u0, int *alphaIndex, cudaStream_t *streams, int ld_x, int ld_u, \
				  T **d_d = nullptr, T *d_dT = nullptr, T *d = nullptr, int ld_d = 0){
	bool defectFlag = M_F > 1 && d_d != nullptr && d_dT != nullptr && d != nullptr && ld_d != 0;
	if (defectFlag){defectKern<<<NUM_ALPHA,NUM_TIME_STEPS,0,streams[0]>>>(d_d,d_dT,ld_d); gpuErrchk(cudaPeekAtLastError());}
	gpuErrchk(cudaMemcpyAsync(x0, h_d_x[*alphaIndex], ld_x*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, streams[1]));
	gpuErrchk(cudaMemcpyAsync(u0, h_d_u[*alphaIndex], ld_u*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, streams[2]));
	gpuErrchk(cudaStreamSynchronize(streams[0]));
	gpuErrchk(cudaMemcpyAsync(d, d_dT, NUM_ALPHA*sizeof(T), cudaMemcpyDeviceToHost, streams[0]));
	gpuErrchk(cudaDeviceSynchronize()); // sync to be done
}

template <typename T>
__host__ __forceinline__
void storeVarsCPU(T *x, T *x0, T *u, T *u0, std::thread *threads, int ld_x, int ld_u, \
				  T *d = nullptr, T *maxd = nullptr, int ld_d = 0){
	bool defectFlag = M_F > 1 && d != nullptr && maxd != nullptr && ld_d != 0;
	if (defectFlag){*maxd = defectComp(d,ld_d);}
	threads[0] = std::thread(memcpy, std::ref(x0), std::ref(x), ld_x*NUM_TIME_STEPS*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
	threads[1] = std::thread(memcpy, std::ref(u0), std::ref(u), ld_u*NUM_TIME_STEPS*sizeof(T));
	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
	threads[0].join();
	threads[1].join();
}

template <typename T>
__host__ __forceinline__
void allocateMemory_GPU(T ***d_x, T ***h_d_x, T **d_xp, T **d_xp2, T ***d_u, T ***h_d_u, T **d_up, T **d_xGoal, T **xGoal, T **d_P, T **d_Pp, T **d_p, T **d_pp, 
						T **d_AB, T **d_H, T **d_g, T **d_KT, T **d_du, T ***d_d, T ***h_d_d, T **d_dp, T **d_dT, T **d_dM, T **d, T **d_ApBK, T **d_Bdu,
                        T **d_JT, T **J, T **d_dJexp, T **dJexp, T **alpha, T **d_alpha, int **alphaIndex, int **d_err, int **err,
						int *ld_x, int *ld_u, int *ld_P, int *ld_p, int *ld_AB, int *ld_H, int *ld_g, int *ld_KT, int *ld_du, int *ld_d, int *ld_A,
                        cudaStream_t **streams, T **d_I = nullptr, T **d_Tbody = nullptr){

	// note on device x,u is [NUM_ALPHA][SIZE*NUM_TIME_STEPS]
	//      on host   x,u is [SIZE*NUM_TIME_STEPS]
	*ld_u = DIM_u_r;	*ld_x = DIM_x_r;	// we should use pitched malloc but for now just set to DIM_<>_r
	*h_d_x = (T **)malloc(NUM_ALPHA*sizeof(T*));
	*h_d_u = (T **)malloc(NUM_ALPHA*sizeof(T*));
	gpuErrchk(cudaMalloc((void**)d_x, NUM_ALPHA*sizeof(T*)));
	gpuErrchk(cudaMalloc((void**)d_u, NUM_ALPHA*sizeof(T*)));
	for (int i=0; i<NUM_ALPHA; i++){
		gpuErrchk(cudaMalloc((void**)&((*h_d_x)[i]),(*ld_x)*NUM_TIME_STEPS*sizeof(T)));
		gpuErrchk(cudaMalloc((void**)&((*h_d_u)[i]),(*ld_u)*NUM_TIME_STEPS*sizeof(T)));
	}
	gpuErrchk(cudaMalloc((void**)d_xp, (*ld_x)*NUM_TIME_STEPS*sizeof(T))); 
	gpuErrchk(cudaMalloc((void**)d_xp2, (*ld_x)*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_up, (*ld_u)*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMemcpy(*d_x, *h_d_x, NUM_ALPHA*sizeof(T*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_u, *h_d_u, NUM_ALPHA*sizeof(T*), cudaMemcpyHostToDevice));

	// and for the xGoal
	int goalSize = EE_COST ? 6 : STATE_SIZE;
	*xGoal = (T *)malloc(goalSize*sizeof(T));
	gpuErrchk(cudaMalloc((void**)d_xGoal,goalSize*sizeof(T)));

	// allocate memory with pitched malloc and thus collect the lds (for now just set to DIM_<>_r)
	*ld_P = DIM_P_r;	*ld_p = DIM_p_r;	*ld_AB = DIM_AB_r;	*ld_H = DIM_H_r;	*ld_g = DIM_g_r;
	*ld_KT = DIM_KT_r;	*ld_du = DIM_du_r;	*ld_d = DIM_d_r;	*ld_A = DIM_A_r;
	gpuErrchk(cudaMalloc((void**)d_AB,(*ld_AB)*DIM_AB_c*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_P,(*ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_Pp,(*ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_p,(*ld_p)*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_pp,(*ld_p)*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_H,(*ld_H)*DIM_H_c*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_g,(*ld_g)*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_KT,(*ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_du,(*ld_du)*NUM_TIME_STEPS*sizeof(T)));
	*h_d_d = (T **)malloc(NUM_ALPHA*sizeof(T*));
	gpuErrchk(cudaMalloc((void**)d_d, NUM_ALPHA*sizeof(T*)));
	for (int i=0; i<NUM_ALPHA; i++){
		gpuErrchk(cudaMalloc((void**)&((*h_d_d)[i]),(*ld_d)*NUM_TIME_STEPS*sizeof(T)));  
	} 
	gpuErrchk(cudaMemcpy(*d_d, *h_d_d, NUM_ALPHA*sizeof(T*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)d_dp, (*ld_d)*NUM_TIME_STEPS*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_dT, NUM_ALPHA*sizeof(T)));
	gpuErrchk(cudaMalloc((void**)d_dM, sizeof(T))); 
	*d = (T *)malloc(NUM_ALPHA*sizeof(T));
	gpuErrchk(cudaMalloc((void**)d_Bdu, (*ld_d)*NUM_TIME_STEPS*sizeof(T)));  
	gpuErrchk(cudaMalloc((void**)d_ApBK, (*ld_A)*DIM_A_c*NUM_TIME_STEPS*sizeof(T)));  

	// then for cost, alpha, rho, and errors
	gpuErrchk(cudaMalloc((void**)d_JT, (EE_COST ? max(M_F*NUM_ALPHA,NUM_TIME_STEPS) : NUM_ALPHA)*sizeof(T)));
	*J = (T *)malloc(NUM_ALPHA*sizeof(T));
	gpuErrchk(cudaMalloc((void**)d_dJexp,2*M_B*sizeof(T)));
	*dJexp = (T *)malloc(2*M_B*sizeof(T));
	*alphaIndex = (int *)malloc(sizeof(int*));
	*alpha = (T *)malloc(NUM_ALPHA*sizeof(T));
	gpuErrchk(cudaMalloc((void**)d_alpha, NUM_ALPHA*sizeof(T)));
	for (int i=0; i<NUM_ALPHA; i++){(*alpha)[i] = pow(ALPHA_BASE,i);}
	gpuErrchk(cudaMemcpy(*d_alpha, *alpha, NUM_ALPHA*sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)d_err,M_B*sizeof(int)));
	*err = (int *)malloc(M_B*sizeof(int));

	// put streams in order of priority
	int priority, minPriority, maxPriority;
	gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
	*streams = (cudaStream_t *)malloc(NUM_STREAMS*sizeof(cudaStream_t));
	for(int i=0; i<NUM_STREAMS; i++){
		priority = min(minPriority+i,maxPriority);
		gpuErrchk(cudaStreamCreateWithPriority(&((*streams)[i]),cudaStreamNonBlocking,priority));
	}

	// load in the Inertia and Tbody if requested
	if (d_I != nullptr){
		T *I = (T *)malloc(36*NUM_POS*sizeof(T));	initI<T>(I);
		gpuErrchk(cudaMalloc((void**)d_I,36*NUM_POS*sizeof(T)));	
		gpuErrchk(cudaMemcpy(*d_I, I, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));	free(I);
	}
	if (d_Tbody != nullptr){
		T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(Tbody);
		gpuErrchk(cudaMalloc((void**)d_Tbody,36*NUM_POS*sizeof(T)));
		gpuErrchk(cudaMemcpy(*d_Tbody, Tbody, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));	free(Tbody);
	}

	// set shared memory banks to T precision
	//gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	// not using cudaFuncSetCacheConfig because documentation says may induce syncs which we don't want

	// sync before exit to make sure it all finished
	gpuErrchk(cudaDeviceSynchronize());
}

template <typename T>
__host__ __forceinline__
void freeMemory_GPU(T **d_x, T **h_d_x, T *d_xp, T *d_xp2, T **d_u, T **h_d_u, T *d_up, T *xGoal, T *d_xGoal, T *d_P, T *d_Pp, T *d_p, T *d_pp, 
					T *d_AB, T *d_H, T *d_g, T *d_KT, T *d_du, T **d_d, T **h_d_d, T *d_dp, T *d_dT, T *d_dM, T *d, T *d_ApBK, T *d_Bdu,
					T *d_JT, T *J, T *d_dJexp, T *dJexp, T *alpha, T *d_alpha, int *alphaIndex, int *d_err, int *err, 
					cudaStream_t *streams, T *d_I = nullptr, T *d_Tbody = nullptr){
		for (int i=0; i<NUM_ALPHA; i++){gpuErrchk(cudaFree(h_d_x[i]));	gpuErrchk(cudaFree(h_d_u[i]));	gpuErrchk(cudaFree(h_d_d[i]));}
		gpuErrchk(cudaFree(d_x));		free(h_d_x);		 	gpuErrchk(cudaFree(d_xp));		gpuErrchk(cudaFree(d_xp2));	
		gpuErrchk(cudaFree(d_u));		free(h_d_u); 			gpuErrchk(cudaFree(d_up));
		gpuErrchk(cudaFree(d_xGoal));	free(xGoal);			
		gpuErrchk(cudaFree(d_P));		gpuErrchk(cudaFree(d_Pp));	gpuErrchk(cudaFree(d_p));	gpuErrchk(cudaFree(d_pp));	
		gpuErrchk(cudaFree(d_AB));		gpuErrchk(cudaFree(d_H));	gpuErrchk(cudaFree(d_g));	gpuErrchk(cudaFree(d_KT));	gpuErrchk(cudaFree(d_du));
		gpuErrchk(cudaFree(d_d));		free(h_d_d);				gpuErrchk(cudaFree(d_dp));	gpuErrchk(cudaFree(d_dT));	gpuErrchk(cudaFree(d_dM));  
		free(d); 						gpuErrchk(cudaFree(d_Bdu));	gpuErrchk(cudaFree(d_ApBK));
		gpuErrchk(cudaFree(d_JT));		free(J);					gpuErrchk(cudaFree(d_dJexp));	free(dJexp);	
		gpuErrchk(cudaFree(d_alpha));	free(alpha);				free(alphaIndex);
		gpuErrchk(cudaFree(d_err));		free(err);	for(int i=0; i<NUM_STREAMS; i++){gpuErrchk(cudaStreamDestroy(streams[i]));}		free(streams);
		if (d_I != nullptr){gpuErrchk(cudaFree(d_I));}	if (d_Tbody != nullptr){gpuErrchk(cudaFree(d_Tbody));}
		gpuErrchk(cudaDeviceSynchronize());
}

template <typename T>
__host__ __forceinline__
void allocateMemory_CPU(T **x, T **xp, T **xp2, T **u, T **up, T **xGoal, T **P, T **Pp, T **p, T **pp, 
						T **AB, T **H, T **g, T **KT, T **du, T **d, T **dp, T **ApBK, T **Bdu, T **JT, T **dJexp, T **alpha, int **err, 
						int *ld_x, int *ld_u, int *ld_P, int *ld_p, int *ld_AB, int *ld_H, int *ld_g, int *ld_KT, int *ld_du, int *ld_d, int *ld_A, 
						T **I = nullptr, T **Tbody = nullptr){
	// allocate memory for x,u with pitched malloc and thus collect the lds (for now just set ld = DIM_<>_r)
	*ld_x = DIM_x_r;	*ld_u = DIM_u_r;
	*x = (T *)malloc((*ld_x)*NUM_TIME_STEPS*sizeof(T));
	*xp = (T *)malloc((*ld_x)*NUM_TIME_STEPS*sizeof(T)); 
	*xp2 = (T *)malloc((*ld_x)*NUM_TIME_STEPS*sizeof(T));
	*u = (T *)malloc((*ld_u)*NUM_TIME_STEPS*sizeof(T));
	*up = (T *)malloc((*ld_x)*NUM_TIME_STEPS*sizeof(T));
	int goalSize = EE_COST ? 6 : STATE_SIZE;
	*xGoal = (T *)malloc(goalSize*sizeof(T));
	// allocate memory for vars with pitched malloc and thus collect the lds (for now just set ld = DIM_<>_r)
	*ld_AB = DIM_AB_r;	*ld_P = DIM_P_r;	*ld_p = DIM_p_r;	*ld_H = DIM_H_r;
	*ld_g = DIM_g_r;	*ld_KT = DIM_KT_r;	*ld_du = DIM_du_r;	*ld_d = DIM_d_r;	*ld_A = DIM_A_r;
	*P = (T *)malloc((*ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
	*Pp = (T *)malloc((*ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
	*p = (T *)malloc((*ld_p)*NUM_TIME_STEPS*sizeof(T));
	*pp = (T *)malloc((*ld_p)*NUM_TIME_STEPS*sizeof(T));
	*AB = (T *)malloc((*ld_AB)*DIM_AB_c*NUM_TIME_STEPS*sizeof(T));
	*H = (T *)malloc((*ld_H)*DIM_H_c*NUM_TIME_STEPS*sizeof(T));
	*g = (T *)malloc((*ld_g)*NUM_TIME_STEPS*sizeof(T));
	*KT = (T *)malloc((*ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
	*du = (T *)malloc((*ld_du)*NUM_TIME_STEPS*sizeof(T));
	*d = (T *)malloc((*ld_d)*NUM_TIME_STEPS*sizeof(T));
	*dp = (T *)malloc((*ld_d)*NUM_TIME_STEPS*sizeof(T));
	*Bdu = (T *)malloc((*ld_d)*NUM_TIME_STEPS*sizeof(T));  
	*ApBK = (T *)malloc((*ld_A)*DIM_A_c*NUM_TIME_STEPS*sizeof(T));  
	// could have FSIM or COST THREADS cost comps
	*JT = (T *)malloc(max(FSIM_THREADS,COST_THREADS)*sizeof(T));
	*dJexp = (T *)malloc(2*M_B*sizeof(T));
	// allocate and init alpha
	*alpha = (T *)malloc(NUM_ALPHA*sizeof(T));
	for (int i=0; i<NUM_ALPHA; i++){(*alpha)[i] = pow(ALPHA_BASE,i);}
	*err = (int *)malloc(M_B*sizeof(int));
	// load in the Inertia and Tbody if requested
	if (I != nullptr){*I = (T *)malloc(36*NUM_POS*sizeof(T));	initI<T>(*I);}
	if (Tbody != nullptr){*Tbody = (T *)malloc(36*NUM_POS*sizeof(T));	initT<T>(*Tbody);}
}

template <typename T>
__host__ __forceinline__
void allocateMemory_CPU2(T ***xs, T **xp, T **xp2, T ***us, T **up, T **xGoal, T **P, T **Pp, T **p, T **pp, 
						T **AB, T **H, T **g, T **KT, T **du, T ***ds, T **dp, T **ApBK, T **Bdu, T ***JTs, T **dJexp, T **alpha, int **err, 
						int *ld_x, int *ld_u, int *ld_P, int *ld_p, int *ld_AB, int *ld_H, int *ld_g, int *ld_KT, int *ld_du, int *ld_d, int *ld_A, 
						T **I = nullptr, T **Tbody = nullptr){
	// allocate the xs and us and ds (and JTs)
	*xs = (T **)malloc(NUM_ALPHA*sizeof(T*));
	*us = (T **)malloc(NUM_ALPHA*sizeof(T*));
	*ds = (T **)malloc(NUM_ALPHA*sizeof(T*));
	*JTs = (T **)malloc(NUM_ALPHA*sizeof(T*));
	allocateMemory_CPU<T>(&((*xs)[0]), xp, xp2, &((*us)[0]), up, xGoal, P, Pp, p, pp, AB, H, g, KT, du, &((*ds)[0]), dp, ApBK, Bdu, &((*JTs)[0]), dJexp, alpha, err, 
						  ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A, I, Tbody);
	for (int i = 1; i < NUM_ALPHA; i++){
		(*xs)[i] = (T *)malloc((*ld_x)*NUM_TIME_STEPS*sizeof(T));
		(*us)[i] = (T *)malloc((*ld_u)*NUM_TIME_STEPS*sizeof(T));
		(*ds)[i] = (T *)malloc((*ld_d)*NUM_TIME_STEPS*sizeof(T));
		(*JTs)[i] = (T *)malloc(max(FSIM_THREADS,COST_THREADS)*sizeof(T));
	}
}

template <typename T>
__host__ __forceinline__
void freeMemory_CPU(T *x, T *xp, T *xp2, T *u, T *up, T *P, T *Pp, T *p, T *pp, T *AB, T *H, T *g, T *KT, T *du, T *d, T *dp, T *Bdu, T *ApBK, 
                    T *dJexp, int *err, T *alpha, T *JT, T *xGoal, T *I = nullptr, T *Tbody = nullptr){
    free(x);    free(xp);   free(xp2);  free(u);    free(up);   free(P);    free(Pp);   free(p);    free(pp);    
    free(AB);   free(H);    free(g);    free(KT);   free(du);   free(d);    free(dp);   free(Bdu);  free(ApBK); 
    free(dJexp);    free(err);  free(alpha);    free(JT);   free(xGoal);    if (I != nullptr){free(I);}    if (Tbody != nullptr){free(Tbody);}
}

template <typename T>
__host__ __forceinline__
void freeMemory_CPU2(T **xs, T *xp, T *xp2, T **us, T *up, T *P, T *Pp, T *p, T *pp, T *AB, T *H, T *g, T *KT, T *du, T **ds, T *dp, T *Bdu, T *ApBK, 
                    T *dJexp, int *err, T *alpha, T **JTs, T *xGoal, T *I = nullptr, T *Tbody = nullptr){
	freeMemory_CPU<T>(xs[0], xp, xp2, us[0], up, P, Pp, p, pp, AB, H, g, KT, du, ds[0], dp, Bdu, ApBK, dJexp, err, alpha, JTs[0], xGoal, I, Tbody);
	for (int i = 1; i < NUM_ALPHA; i++){
		free(xs[i]);	free(us[i]);	free(ds[i]);	free(JTs[i]);
	}
	free(xs); free(us); free(ds); free(JTs);
}
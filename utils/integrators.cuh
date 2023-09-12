/*****************************************************************
 * Integrator(Gradient) Helper Functions
 * (currently only supports iLQR - UDP in future release)
 *
 *  1: Euler
 *  2: Midpoint
 *  3: Rk3
 *
 *  _integrator
 *    dynamics (from dynamics)
 *  _integratorGradient
 *    dynamicsGradient (from dynamics)
 ****************************************************************/
// x = [q;qd] so xd = [dq,qdd] thus dxd_dx,u = [0_{numpos},I_{numpos},0_{numpos};dqdd]
 template <typename T>
	__host__ __device__ __forceinline__
T dqdd2dxd(T *dqdd, int r, int c){return r < NUM_POS ? static_cast<T>(r + NUM_POS == c ? 1 : 0) : dqdd[(c-1)*NUM_POS + r];}

template <typename T>
	__host__ __device__ __forceinline__
T dqddk2dxd(T dqddk, int r, int c){return r < NUM_POS ? static_cast<T>(r + NUM_POS == c ? 1 : 0) : dqddk;}

#if INTEGRATOR == 1 // Euler
	template <typename T>
	__host__ __device__ __forceinline__
	void _integrator(T *s_xkp1, T *s_x, T *s_u, T *s_qdd, T *d_I, T *d_Tbody, T dt, T *s_eePos = nullptr, T *s_eeVel = nullptr){
		int start, delta; singleLoopVals(&start,&delta);
		// compute the new state by first computing dynamics
		dynamics<T>(s_qdd,s_x,s_u,d_I,d_Tbody,s_eePos,1,s_eeVel);
		hd__syncthreads();
		// then use the euler rule
		for (int ind = start; ind < NUM_POS; ind += delta){
			s_xkp1[ind] = s_x[ind] + dt*s_x[ind+NUM_POS];
			s_xkp1[ind+NUM_POS] = s_x[ind+NUM_POS] + dt*s_qdd[ind];
		}
	}

	template <typename T>
	__host__ __device__ __forceinline__
	void _integratorGradient(T *ABk, T *s_x, T *s_u, T *s_qdd, T *s_dqdd, T *d_I, T *d_Tbody, T dt, int ld_AB){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		// first compute the dynamics gradient
		dynamicsGradient<T>(s_dqdd,s_qdd,s_x,s_u,d_I,d_Tbody);  
		hd__syncthreads();
		// then apply the euler rule -- xkp1 = xk + h*dxk thus AB = [I_{state},0_{control}] + h*dxd
		#pragma unroll
		for (int ky = starty; ky < DIM_AB_c; ky += dy){ // pick the col dq, dqdd, du
			#pragma unroll
			for (int kx = startx; kx < DIM_AB_r; kx += dx){ // pick the row (q,dq)
				ABk[ky*ld_AB + kx] = static_cast<T>(ky == kx ? 1 : 0) + dt*dqdd2dxd(s_dqdd,kx,ky);
			}
		}
	}

#elif INTEGRATOR == 2 // midpoint
		template <typename T>
	__host__ __device__ __forceinline__
	void _integrator(T *s_xkp1, T *s_x, T *s_u, T *s_qdd, T *d_I, T *d_Tbody, T dt, T *s_eePos = nullptr, T *s_eeVel = nullptr){
		#ifdef __CUDA_ARCH__
			__shared__ T s_xm[STATE_SIZE_PDDP];
		#else
			T s_xm[STATE_SIZE_PDDP];
		#endif
		int start, delta; singleLoopVals(&start,&delta);
		// first compute dynamics at the initial point
		dynamics<T>(s_qdd,s_x,s_u,d_I,d_Tbody,s_eePos,1,s_eeVel);
		hd__syncthreads();
		// then compute middle point and compute dynamics there
		for (int ind=start; ind<NUM_POS; ind+=delta){
			s_xm[ind] = 		s_x[ind] 		 + static_cast<T>(0.5)*dt*s_x[ind+NUM_POS];
			s_xm[ind+NUM_POS] = s_x[ind+NUM_POS] + static_cast<T>(0.5)*dt*s_qdd[ind];
		}
		hd__syncthreads();
		dynamics<T>(s_qdd,s_xm,s_u,d_I,d_Tbody);
		hd__syncthreads();
		// then use the final integration rule
		for (int ind = start; ind < NUM_POS; ind += delta){
			s_xkp1[ind] 		= s_x[ind] 		   + dt*s_x[ind+NUM_POS];
			s_xkp1[ind+NUM_POS] = s_x[ind+NUM_POS] + dt*s_qdd[ind];
		}
	}
	

	template <typename T>
	__host__ __device__ __forceinline__
	void _integratorGradient(T *ABk, T *s_x, T *s_u, T *s_qdd, T *s_dqdd, T *d_I, T *d_Tbody, T dt, int ld_AB){
		#ifdef __CUDA_ARCH__
			__shared__ T s_xm[STATE_SIZE_PDDP];	__shared__ T s_qdd2[NUM_POS];	__shared__ T s_dqdd2[NUM_POS*(STATE_SIZE_PDDP+CONTROL_SIZE)];
		#else
			T s_xm[STATE_SIZE_PDDP];		T s_qdd2[NUM_POS];		T s_dqdd2[NUM_POS*(STATE_SIZE_PDDP+CONTROL_SIZE)];
		#endif
		int start, delta; singleLoopVals(&start,&delta);
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		// first compute dynamics and grad at the initial point
		dynamicsGradient<T>(s_dqdd,s_qdd,s_x,s_u,d_I,d_Tbody);		hd__syncthreads();
		// then compute middle point and compute dynamics there
		for (int ind=start; ind<NUM_POS; ind+=delta){
			s_xm[ind] = 		s_x[ind] 		 + static_cast<T>(0.5)*dt*s_x[ind+NUM_POS];
			s_xm[ind+NUM_POS] = s_x[ind+NUM_POS] + static_cast<T>(0.5)*dt*s_qdd[ind];
		}
		hd__syncthreads();
		// then compute the dynamics gradients at middle point
		dynamicsGradient<T>(s_dqdd2,s_qdd2,s_xm,s_u,d_I,d_Tbody);  hd__syncthreads();
		// then apply the midpoint rule ABX = [I_{state},0_{control}] + h/2*[0_{numpos},I_{numpos},0_{numpos};dqdd]
		// and AB = A2*AB1 + [0,B2] thus AX = I + h/2*[0,I;dqdd_dx] and BX = h/2*[0;dqdd_du]
		#pragma unroll
		for (int ky = starty; ky < DIM_AB_c; ky += dy){ // pick the col dq, dqdd, du
			#pragma unroll
			for (int kx = startx; kx < DIM_AB_r; kx += dx){ // pick the row (q,dq)
				T val = 0;
				for (int i = 0; i < DIM_AB_r; i++){
					T A2_val = static_cast<T>(kx == i ? 1 : 0) + static_cast<T>(0.5)*dt*dqdd2dxd(s_dqdd2,kx,i);
					T AB1_val = static_cast<T>(ky == i ? 1 : 0) + static_cast<T>(0.5)*dt*dqdd2dxd(s_dqdd,i,ky);
					val += A2_val * AB1_val;
				}
				// then add in the [0,B2] and save
				ABk[ky*ld_AB + kx] = val + (ky < STATE_SIZE_PDDP ? static_cast<T>(0) : static_cast<T>(0.5)*dt*dqdd2dxd(s_dqdd2,kx,ky));
			}
		}
	}

#elif INTEGRATOR == 3 // RK3
	template <typename T>
	__host__ __device__ __forceinline__
	void _integrator(T *s_xkp1, T *s_x, T *s_u, T *s_qdd1, T *d_I, T *d_Tbody, T dt, T *s_eePos = nullptr, T *s_eeVel = nullptr){
		#ifdef __CUDA_ARCH__
			__shared__ T s_x2[STATE_SIZE_PDDP];    __shared__ T s_x3[STATE_SIZE_PDDP];
			__shared__ T s_qdd2[NUM_POS];     __shared__ T s_qdd3[NUM_POS];
		#else
			T s_x2[STATE_SIZE_PDDP];	T s_x3[STATE_SIZE_PDDP];	T s_qdd2[NUM_POS];	T s_qdd3[NUM_POS];
		#endif
		int start, delta; singleLoopVals(&start,&delta);
		// first compute dynamics at the initial point
		dynamics<T>(s_qdd1,s_x,s_u,d_I,d_Tbody,s_eePos,1,s_eeVel);
		hd__syncthreads();
		// then compute middle point and compute dynamics there
		for (int ind=start; ind<NUM_POS; ind+=delta){
			s_x2[ind] 		  =	s_x[ind] 		 + static_cast<T>(0.5)*dt*s_x[ind+NUM_POS];
			s_x2[ind+NUM_POS] = s_x[ind+NUM_POS] + static_cast<T>(0.5)*dt*s_qdd1[ind];
		}
		hd__syncthreads();
		dynamics<T>(s_qdd2,s_x2,s_u,d_I,d_Tbody);
		hd__syncthreads();
		// then compute third point and compute dynamics there
		for (int ind=start; ind<NUM_POS; ind+=delta){
			s_x3[ind] 		  = s_x[ind] 		 + dt*(static_cast<T>(2)*s_x2[ind+NUM_POS] - s_x[ind+NUM_POS]);
			s_x3[ind+NUM_POS] = s_x[ind+NUM_POS] + dt*(static_cast<T>(2)*s_qdd2[ind] 	   - s_qdd1[ind]);
		}
		hd__syncthreads();
		dynamics<T>(s_qdd3,s_x3,s_u,d_I,d_Tbody);
		hd__syncthreads();
		// then use the final integration rule
		for (int ind = start; ind < NUM_POS; ind += delta){
			s_xkp1[ind] 		= s_x[ind] 		   + (dt/static_cast<T>(6))*(s_x[ind+NUM_POS] + static_cast<T>(4)*s_x2[ind+NUM_POS] + s_x3[ind+NUM_POS]);
			s_xkp1[ind+NUM_POS] = s_x[ind+NUM_POS] + (dt/static_cast<T>(6))*(s_qdd1[ind] 	  + static_cast<T>(4)*s_qdd2[ind]       + s_qdd3[ind]);
		}
	}

	template <typename T>
	__host__ __device__ __forceinline__
	void _integratorGradient(T *ABk, T *s_x, T *s_u, T *s_qdd_1, T *s_dqdd_1, T *d_I, T *d_Tbody, T dt, int ld_AB){
		int start, delta; singleLoopVals(&start,&delta);
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		#ifdef __CUDA_ARCH__
			__shared__ T s_TMP_1[DIM_AB_r*DIM_AB_c];		__shared__ T s_xm_1[STATE_SIZE_PDDP];
			__shared__ T s_TMP_2[DIM_AB_r*DIM_AB_c];		__shared__ T s_xm_2[STATE_SIZE_PDDP];
			__shared__ T s_qdd_2[NUM_POS];					__shared__ T s_dqdd_2[NUM_POS*(STATE_SIZE_PDDP+CONTROL_SIZE)];
			__shared__ T s_qdd_3[NUM_POS];					__shared__ T s_dqdd_3[NUM_POS*(STATE_SIZE_PDDP+CONTROL_SIZE)];
		#else
			 T s_TMP_1[DIM_AB_r*DIM_AB_c];		 T s_xm_1[STATE_SIZE_PDDP];
			 T s_TMP_2[DIM_AB_r*DIM_AB_c];		 T s_xm_2[STATE_SIZE_PDDP];
			 T s_qdd_2[NUM_POS];				 T s_dqdd_2[NUM_POS*(STATE_SIZE_PDDP+CONTROL_SIZE)];
			 T s_qdd_3[NUM_POS];			     T s_dqdd_3[NUM_POS*(STATE_SIZE_PDDP+CONTROL_SIZE)];
		#endif
		// first get the three gradients (start at the start)
		dynamicsGradient<T>(s_dqdd_1,s_qdd_1,s_x,s_u,d_I,d_Tbody);
	   	hd__syncthreads();
	   	// then get xm_1 and compute gradient there
		#pragma unroll
	   	for (int i=start; i<NUM_POS; i += delta){
	      	s_xm_1[i] 			= s_x[i] + static_cast<T>(0.5)*dt*s_x[i+NUM_POS];
	      	s_xm_1[i + NUM_POS] = s_x[i] + static_cast<T>(0.5)*dt*s_qdd_1[i];
	   	}
	   	hd__syncthreads();
	   	dynamicsGradient<T>(s_dqdd_2,s_qdd_2,s_xm_1,s_u,d_I,d_Tbody);
	   	hd__syncthreads();
	   	// then get xm2 and compute the gradient there
		#pragma unroll
	   	for (int i=start; i<NUM_POS; i += delta){
	   		s_xm_2[i] 			= s_x[i] + dt*s_x[i+NUM_POS] + static_cast<T>(2)*dt*s_xm_1[i+NUM_POS];
	      	s_xm_2[i + NUM_POS] = s_x[i] + dt*s_qdd_1[i] 	 + static_cast<T>(2)*dt*s_qdd_2[i];
	   	}
	   	hd__syncthreads();
	   	dynamicsGradient<T>(s_dqdd_3,s_qdd_3,s_xm_2,s_u,d_I,d_Tbody);
		hd__syncthreads();
		// TMP_1  = [0,B2] + A2*([I,0] + (h/2)*AB1); where ABX = [dqdX;dqddX] = [0,I,0;dqddX]
		#pragma unroll
		for (int ky = starty; ky < DIM_AB_c; ky += dy){
			#pragma unroll
			for (int kx = startx; kx < DIM_AB_r; kx += dx){
				T val = 0;
				#pragma unroll
				for (int i=0; i<DIM_AB_r; i++){
					val += dqdd2dxd(s_dqdd_2,kx,i)*(static_cast<T>(0.5)*dt*dqdd2dxd(s_dqdd_1,i,ky) + static_cast<T>(ky == i ? 1 : 0));
				}
				s_TMP_1[kx + DIM_AB_r*ky] = val + (ky < STATE_SIZE_PDDP ? static_cast<T>(0) : dqdd2dxd(s_dqdd_2,kx,ky));
			}
		}
		hd__syncthreads();
		// TMP_2 = [0,B3] + A3*([I,0] + 2*h*TMP1 - h*AB1);
		#pragma unroll
		for (int ky = starty; ky < DIM_AB_c; ky += dy){
			#pragma unroll
			for (int kx = startx; kx < DIM_AB_r; kx += dx){
				T val = 0;
				#pragma unroll
				for (int i=0; i<DIM_AB_r; i++){
					val += dqdd2dxd(s_dqdd_3,kx,i)*(static_cast<T>(2)*dt*s_TMP_1[ky*DIM_AB_r + i] - dt*dqdd2dxd(s_dqdd_1,i,ky) + static_cast<T>(ky == i ? 1 : 0));
	          		}
	          		s_TMP_2[kx + DIM_AB_r*ky] = val + (ky < STATE_SIZE_PDDP ? static_cast<T>(0) : dqdd2dxd(s_dqdd_3,kx,ky));
			}
		}
		hd__syncthreads();
		// AB = [I,0] + (h/6)*AB1 + (2*h/3)*TMP1 + (h/6)*TMP2;
		#pragma unroll
		for (int ky = starty; ky < DIM_AB_c; ky += dy){
			#pragma unroll
			for (int kx = startx; kx < DIM_AB_r; kx += dx){
				ABk[kx + ld_AB*ky] = (dt/static_cast<T>(6))*dqdd2dxd(s_dqdd_1,kx,ky) + (static_cast<T>(2)*dt/static_cast<T>(3))*s_TMP_1[kx + DIM_AB_r*ky] +
     								 (dt/static_cast<T>(6))*s_TMP_2[kx + DIM_AB_r*ky] + static_cast<T>(kx == ky ? 1 : 0);
			}
		}
	}
#else
	#error "Currently only supports Euler[1], Midpoint[2], or RK[3].\n"
#endif
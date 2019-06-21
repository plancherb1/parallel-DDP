/*****************************************************************
 * DDP Backward Pass Helper Functions
 * (currently only supports iLQR - UDP in future release)
 *
 *  backwardPass(C/G)PU
 *    backPass(Threaded/Kern)
 *      linearXfrmOrLoad
 *    backprop
 *    computeKTdu_dim1 OR ((invHuu_dim4 OR invHuu) AND computeKTdu)
 *    computeCTG
 *    computeFSVars
 *    computeExpRed
 *****************************************************************/

// load in the Pp matricies and compute the linear transform
template <typename T>
__host__ __device__ __forceinline__
void linearXfrmOrLoad(T *s_P, T *s_p, T *s_dx, T *d_x, T *d_xp, int ks, int ld_x, int ld_P, int LIN_XFRM_FLAG, T *b_P = nullptr, T *b_p = nullptr){
	int start, delta; singleLoopVals(&start,&delta);
	// in both cases start by computing s_dx if needed
	if (LIN_XFRM_FLAG){loadDeltaV<T,DIM_x_r>(s_dx, ld_x*(ks+1) + d_x, ld_x*(ks+1) + d_xp);}
	// if on GPU do shared mem loading and compute
	#ifdef  __CUDA_ARCH__
		// load in initial P and sync
		loadMatToShared<T,DIM_P_r,DIM_P_c>(s_P,b_P,ld_P);
		__syncthreads();
		// load in p and compute linxfrm if needed (s_p += s_P*s_dx) else just set s_p = b_p
		if (LIN_XFRM_FLAG){matVMult<T,DIM_p_r,DIM_x_r>(s_p,s_P,DIM_P_r,s_dx,1.0,b_p);}
		else{loadMatToShared<T,DIM_p_r,DIM_p_c>(s_p,b_p,1);}
	// if on CPU then simply compute the transform on the data in place if needed else already done
	#else
		if (LIN_XFRM_FLAG){matVMult<T,DIM_p_r,DIM_x_r,1>(s_p,s_P,DIM_P_r,s_dx);}
	#endif
}

// BACKPROP CTG store in s_H and s_g uses s_AB, s_AB2, s_P, s_p, b_H, b_g, b_AB, b_d, rho
template <typename T>
__host__ __device__ __forceinline__
void backprop(T *s_H, T *s_g, T *s_AB, T *s_AB2, T *s_P, T *s_p, \
              T *b_d, T rho, int iter, int ld_H, int ld_AB, int ld_P, \
              T *b_H = nullptr, T *b_g = nullptr, T *b_AB = nullptr){ // 3rd row optional arguments needed on the GPU but not on the CPU
	// only double loops here so lets get the start and deltas
	int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
  	// first read in AB to shared memory if on GPU else already in "shared"
  	#ifdef  __CUDA_ARCH__
  		loadMatToShared<T,DIM_AB_r,DIM_AB_c>(s_AB,b_AB,ld_AB);
  		__syncthreads();
	#endif
	// then Propagate cost-to-go function through linearized dynamics for the Hessian and gradient
	// H = [A B]'*P*[A B] + H;      // g = [A B]'*[p + Pd] + g;
	// using the Tassa style regularization we end up with
	// H = [A B]'*(P + rho*diag(in xx block of comp ? 0 : 1))*[A B] + H;      // g = [A B]'*[p + (P + diag(in x block of comp ? 0 : 1))d] + g;
	// First we need to compute AB2 = AB'*(P + rho*diag(in u block));
  	#pragma unroll
  	for (int ky = starty; ky < DIM_ABT_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_ABT_r; kx += dx){ // multiply column ky of P by row kx of AB' == column of AB place in (kx,ky) -> ky*DIM_ABT_r+kx of AB2
      		T val = 0;
      		#pragma unroll
      		for (int j=0; j < DIM_P_c; j++){
        		// note P = P + diag(rho) if in P*B
        		val += s_AB[kx*DIM_AB_r + j] * (s_P[ky*DIM_P_r + j] + ((STATE_REG && kx >= DIM_x_r && ky == j) ? rho : static_cast<T>(0))); 
      		}
      		s_AB2[ky*DIM_ABT_r + kx] = val;
    	}
  	}
 	// And we need to compute p += (P (+ diag(rho) if in P*B))*d if more than 1 shooting interval and on defect boundary
  	#pragma unroll
  	for (int ky = starty; ky < DIM_p_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_p_r; kx += dx){ // dim_p_c == 1 so just limits to one set of threads
     		T val = 0;
      		if (M_BLOCKS_F > 1 && onDefectBoundary(iter)){
        		#pragma unroll
        		for (int j=0; j < DIM_p_r; j++){
          			val += b_d[j] * (s_P[kx + j*DIM_P_r] + ((STATE_REG && kx >= DIM_x_r && kx == j) ? rho : static_cast<T>(0))); // multiply a row of P by d
        		}
      		}
      		s_p[kx] += val;
    	}
  	}
  	// We can then compute and save to shared memory H += AB2*AB 
	// We can also compute and save to shared memory g += AB'*p
	hd__syncthreads();
  	#ifdef  __CUDA_ARCH__
  		matMult<T,DIM_H_r,DIM_H_c,DIM_AB_r>(s_H,DIM_H_r,s_AB2,DIM_ABT_r,s_AB,DIM_AB_r,1.0,b_H,ld_H); // (storing in s_H adding from b_H)
  		matMult<T,DIM_g_r,DIM_g_c,DIM_p_r>(s_g,1,s_p,1,s_AB,DIM_AB_r,1.0,b_g,1); // tricking it to use AB' // (storing in s_g adding from b_g) 
  																				 
	#else
  		matMult<T,DIM_H_r,DIM_H_c,DIM_AB_r,1>(s_H,ld_H,s_AB2,DIM_ABT_r,s_AB,DIM_AB_r); // += s_H
  		matMult<T,DIM_g_r,DIM_g_c,DIM_p_r,1>(s_g,1,s_p,1,s_AB,DIM_AB_r); // tricking it to use AB' // += s_g
	#endif
}

// COMPUTE KT and du store in s_K, s_du, b_KT, b_du and use s_H, (s_Huu, s_Huu2), s_g, d_err, rho to compute
template <typename T>
__host__ __device__ __forceinline__
int computeKTdu_dim1(T *s_K, T *s_du, T *s_H, T *s_g, T rho, int ld_KT, \
					 T *b_KT = nullptr, T *b_du = nullptr){
  	// we need to make sure that s_H is > 0
  	if (s_H[OFFSET_HUU] <= static_cast<T>(0)){
    	return 1;
  	}
  	T val = static_cast<T>(1)/(s_H[OFFSET_HUU] + (!STATE_REG ? rho : static_cast<T>(0))); // load into each thread
  	// then multiply through to the gu and hux blocks (computing K and du --> store to global memory)
  	int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
  	#pragma unroll
  	for (int ky = starty; ky < DIM_Hux_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_Hux_r; kx += dx){ // Huu_dim1 means DIM_Hux_r == 1 so only ky changes
    		s_K[kx + ky*DIM_K_r] = s_H[OFFSET_HUX_GU + kx + ky*DIM_H_r] * val;
    		// #ifdef  __CUDA_ARCH__
	      		b_KT[ky + kx*ld_KT] = s_K[kx + ky*DIM_K_r]; // note the transpose
      		// #endif
    	}
  	} 
  	#pragma unroll
  	for (int ky = starty; ky < DIM_gu_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_gu_r; kx += dx){ // dim_gu_c == 1 so just limits to one set of threads
    		s_du[kx] = s_g[OFFSET_HUX_GU + kx] * val;
    		// #ifdef __CUDA_ARCH__
      			b_du[kx] = s_du[kx];
  			// #endif
    	}
  	}
  	return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int invHuu_dim4(T *s_H, T *s_Huu, T rho){
	// First we need to invert Huu we can do this using the adjugate formula
	// so first load into Huu for manipulations and regularize if needed
	T *s_Huu2 = &s_Huu[DIM_Huu_r*DIM_Huu_c];
	if (!STATE_REG){loadAndRegToShared<T,DIM_Huu_r,DIM_Huu_c>(s_Huu2,&s_H[OFFSET_HUU],DIM_H_r,rho);}
	else{loadMatToShared<T,DIM_Huu_r,DIM_Huu_c>(s_Huu2,&s_H[OFFSET_HUU],DIM_H_r);}
	hd__syncthreads();
  	int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
  	// we need to solve the co-factor matricies in parallel and we know they are all 3x3s
  	// we load in the 3 rows and 3 columns not r=kx and c=ky (unrolled below)
  	#pragma unroll
  	for (int ky = starty; ky < DIM_Huu_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_Huu_r; kx += dx){
      		T CFM[9];
			int r0 = (kx+1)%4;
			int c0 = (ky+1)%4;
			int r1 = (r0+1)%4;
			int c1 = (c0+1)%4;
			int r2 = (r1+1)%4;
			int c2 = (c1+1)%4;
			CFM[0] = s_Huu2[c0*DIM_Huu_r+r0]; //r=0,c=0
			CFM[1] = s_Huu2[c0*DIM_Huu_r+r1]; //r=1,c=0
			CFM[2] = s_Huu2[c0*DIM_Huu_r+r2];
			CFM[3] = s_Huu2[c1*DIM_Huu_r+r0]; 
			CFM[4] = s_Huu2[c1*DIM_Huu_r+r1]; 
			CFM[5] = s_Huu2[c1*DIM_Huu_r+r2];
			CFM[6] = s_Huu2[c2*DIM_Huu_r+r0]; 
			CFM[7] = s_Huu2[c2*DIM_Huu_r+r1]; 
			CFM[8] = s_Huu2[c2*DIM_Huu_r+r2];
			// then compute the determinant of each cofactor matrix (use the 3x3 trick)
			T cdet = CFM[0]*CFM[4]*CFM[8] + CFM[3]*CFM[7]*CFM[2] + CFM[6]*CFM[1]*CFM[5] - \
			       	 CFM[2]*CFM[4]*CFM[6] - CFM[5]*CFM[7]*CFM[0] - CFM[8]*CFM[1]*CFM[3];
      		// and compute the sign for the adjoint
			T csgn = (kx+ky) % 2 ? -1 : 1;
			// then load into the shared mem matrix
			s_Huu[ky*DIM_Huu_r+kx] = csgn*cdet;
    	}
  	}
  	hd__syncthreads();
  	// we now have loaded in all of the adjugates and can easily compute the determinant and final inverse (1/det * adjugates^T)
  	// first compute the det by multiplying the adjugates with the the values in one row
  	T val = static_cast<T>(1)/(s_Huu[0]*s_Huu2[0] + s_Huu[1]*s_Huu2[1] + s_Huu[2]*s_Huu2[2] + s_Huu[3]*s_Huu2[3]);
  	// test for pd matrix
  	if (val <= static_cast<T>(0)){ // failure
    	return 1;
  	}
  	// then multiply through by the det in parallel and transpose into 2nd half of Huu
  	#pragma unroll
  	for (int ky = starty; ky < DIM_Huu_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_Huu_r; kx += dx){
      		s_Huu2[kx*DIM_Huu_r+ky] = val * s_Huu[ky*DIM_Huu_r+kx];
    	}
  	}
  	return 0;
}

template <typename T, int SPEED = 0>
__host__ __device__ __forceinline__
int invHuu(T *s_H, T *s_Huu, T rho, T *s_temp = nullptr){
  	// First we need to invert Huu we can do this using the adjugate formula
  	// so first load into Huu for manipulations as [Huu | I] and regularize if needed
  	if (!STATE_REG){loadAndRegToShared<T,DIM_Huu_r,DIM_Huu_c>(s_Huu,&s_H[OFFSET_HUU],DIM_H_r,rho);}
  	else{loadMatToShared<T,DIM_Huu_r,DIM_Huu_c>(s_Huu,&s_H[OFFSET_HUU],DIM_H_r);}
  	loadIdentity<T,DIM_Huu_r,DIM_Huu_c>(&s_Huu[DIM_Huu_c*DIM_Huu_r],DIM_Huu_r);
  	hd__syncthreads();
  	// then invert it and check for error
  	if (invertMatrix<T,NUM_POS,SPEED>(s_Huu,s_temp)){
    	return 1;
  	}
  	return 0;
}

template <typename T>
__host__ __device__ __forceinline__
void computeKTdu(T *s_K, T *s_du, T *s_H, T *s_g, T *s_Huu, int ld_KT, \
				 T *b_KT = nullptr, T *b_du = nullptr){
  	// Then we can compute K = invHuu*Hux -- not sure why we need to do backwards and invert but that seems to be the only way right now
 	matMult<T,DIM_K_c,DIM_K_r,DIM_Huu_r,0,1>(s_K,DIM_K_r,s_Huu,DIM_Huu_r,&s_H[OFFSET_HUX_GU],DIM_H_r);
  	// We can then compute du = invHuu*gu
  	matVMult<T,DIM_du_r,DIM_Huu_r>(s_du,s_Huu,DIM_Huu_r,&s_g[OFFSET_HUX_GU]);
  	hd__syncthreads();
  	// #ifdef __CUDA_ARCH__
	  	// make sure to save to both shared (for rest of backpass) and gloabl (for forwardpass) memory
	  	saveMatFromSharedT<T,DIM_KT_r,DIM_KT_c>(b_KT,s_K,ld_KT);
	  	saveMatFromShared<T,DIM_du_r,DIM_du_c>(b_du,s_du,1);
	// #endif
}

// COMPUTE CTG store in b_Pprev, b_pprev, s_P, s_p, compute from S_H, s_g, s_K, s_du, s_AB2 
template <typename T>
__host__ __device__ __forceinline__
void computeCTG(T *s_P, T *s_p, T *s_H, T *s_g, T *s_K, T *s_du, T *s_AB2, \
                int iter, int ld_P, T *b_Pprev = nullptr, T *b_pprev = nullptr){
  	// we need to compute: pprev = gx + K'*Huu*du - Hxu*du - K'*gu || Pprev = Hxx + K'*Huu*K - Hxu*K  - K'*Hux;
  	// first we need to compute K'*Huu - Hxu -- note s_AB2 is now unused and can be reused so we save there
  	// so multiply column ky of Huu by row kx of K' = column kx of K and subtract (kx,ky) of Hxu and store in (kx,ky) of s_AB2
	int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		#pragma unroll
		for (int ky = starty; ky < DIM_KT_c; ky += dy){
		#pragma unroll
		for (int kx = startx; kx < DIM_KT_r; kx += dx){
			T val = STATE_REG ? dotProd<T,DIM_K_r>(&s_K[kx*DIM_K_r],1,&s_H[OFFSET_HUU + ky * DIM_H_r],1) : static_cast<T>(0);
			s_AB2[kx + ky * DIM_KT_r] = val - s_H[OFFSET_HXU + kx + DIM_H_r * ky];
  		}
	}
  	hd__syncthreads();
  	// So now we can compute Pprev = Hxx + s_AB2*K - K'*Hux \approx Hxx - Hxu*K
  	// multiply c_ky K * (r_kx s_AB2 - r_kx Hxu) - c_kx of K by c_ky of Hux subtract that from (kx,ky) of Hxx and place in (kx,ky) of Pnm1
  	#pragma unroll
  	for (int ky = starty; ky < DIM_P_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_P_r; kx += dx){
      		T val = 0;
			#pragma unroll
			for (int j=0; j < DIM_K_r; j++){
				val += s_AB2[kx + DIM_KT_r * j] * s_K[ky * DIM_K_r + j] - (STATE_REG ? s_K[kx * DIM_K_r + j] * s_H[OFFSET_HUX_GU + ky * DIM_H_r + j] : static_cast<T>(0));
			}
      		#ifdef __CUDA_ARCH__
      			s_P[kx+ky*DIM_P_r] = s_H[kx+ky*DIM_H_r] + val;
      			b_Pprev[kx+ky*ld_P] = s_P[kx+ky*DIM_P_r]; // if on GPU make sure to save to global mem
  			#else
      			s_P[kx+ky*ld_P] = s_H[kx+ky*DIM_H_r] + val; // else s_P is b_Pprev its all global on CPU
  			#endif
		}
  	}
  	// So now we can compute pprev = gx + s_AB2*du - K'*gu \approx gx - Hxu*du
  	// so we can proceeed like above: c_ky du * (r_kx s_AB2 - r_kx Hxu) - c_kx K * c_ky gu
  	#pragma unroll
  	for (int ky = starty; ky < DIM_p_c; ky += dy){
    	#pragma unroll
    	for (int kx = startx; kx < DIM_p_r; kx += dx){ // note DIM_p_c == 1
      		T val = 0;
      		#pragma unroll
      		for (int j=0; j < DIM_du_r; j++){
        		val += s_du[j] * s_AB2[kx + DIM_KT_r * j] - (STATE_REG ? s_K[kx * DIM_K_r + j] * s_g[OFFSET_HUX_GU + j] : static_cast<T>(0));
      		}
      		s_p[kx] = s_g[kx] + val;
      		#ifdef __CUDA_ARCH__
      			b_pprev[kx] = s_p[kx]; // if on GPU make sure to save to global mem
  			#endif
    	}
  	}
}

// COMPUTE vars for Forward Sweep store in b_ApBK, b_Bdu, compute from s_AB, s_K, s_du
template <typename T>
__host__ __device__ __forceinline__
void computeFSVars(T *b_ApBK, T *b_Bdu, T *s_AB, T *s_K, T *s_du, int ld_A){
  	// ApBK (A - B*K) and Bdu (-B*du)
	// matMult<T,DIM_A_r,DIM_A_c,DIM_K_r>(b_ApBK,ld_A,&s_AB[OFFSET_B],DIM_AB_r,s_K,DIM_K_r,-1.0,s_AB,DIM_AB_r);
	// matVMult<T,DIM_d_r,DIM_du_r>(b_Bdu,&s_AB[OFFSET_B],DIM_AB_r,s_du);
	int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
	#pragma unroll
	for (int ky = starty; ky < DIM_A_c; ky += dy){
		#pragma unroll
		for (int kx = startx; kx < DIM_A_r; kx += dx){
			T val = 0;
			#pragma unroll
			for (int j=0; j<DIM_u_r; j++){
				// multiply row kx of B by column ky of K store in (kx,ky) of XM
				val += s_AB[OFFSET_B + kx + DIM_AB_r*j]*s_K[ky*DIM_K_r + j];
			}
			b_ApBK[kx + ld_A*ky] = s_AB[kx + DIM_AB_r*ky] - val; // dont forget to add (kx,ky) of A
		}
	}
	#pragma unroll
	for (int ky = starty; ky < DIM_d_c; ky += dy){
		#pragma unroll
		for (int kx = startx; kx < DIM_d_r; kx += dx){
			T val = 0;
			#pragma unroll
			for (int j=0; j<DIM_du_r; j++){
				// multiply row kx of B by column ky of du store in (kx,ky) of d
				val += s_AB[OFFSET_B + kx + DIM_AB_r*j]*s_du[j];
			}
			b_Bdu[kx] = val;
		}
	}
}

// COMPUTE expected cost reduction store in s_dJ, compute from s_H, s_g, and s_du
template <typename T>
__host__ __device__ __forceinline__
void computeExpRed(T *s_dJ, T *s_H, T *s_g, T *s_du){
  	// dJexp[2*i] = SUM du'*gu -- note both are vectors so dot product
  	// dJexp[2*i+1] = SUM du'*Huu*du so first compute Huu*du so multiply column ky of du by row kx of Huu and then du'*val (dot)
	int start, delta; singleLoopVals(&start,&delta);
  	#pragma unroll
  	for (int ind = start; ind < DIM_du_r; ind += delta){
  		T val1 = s_du[ind]*s_g[OFFSET_HUX_GU+ind];
  		T val2 = s_du[ind]*dotProd<T,DIM_du_r>(&s_H[OFFSET_HUU+ind],DIM_H_r,s_du,1);
  		// if on cuda we are doing this in parallel if not it's serial so just store serially
  		#ifdef __CUDA_ARCH__
      		s_dJ[ind] 			 += val1;
      		s_dJ[DIM_du_r + ind] += val2;
  		#else
  			  s_dJ[0] += val1; 
      		s_dJ[1] += val2;
  		#endif
  	}
}

//<<<M_BLOCKS_B,Dim3(ld_H,ld_H)>>>
template <typename T>
__global__
void backPassKern(T *d_AB, T *d_P, T *d_p, T *d_Pp, T *d_pp, T *d_H, \
                  T *d_g, T *d_KT, T *d_du, T *d_d, T *d_ApBK, T *d_Bdu, \
                  T *d_x, T *d_xp, T *d_dJexp, int *d_err, T rho, \
                  int ld_AB, int ld_P, int ld_p, int ld_H, int ld_g, \
                  int ld_KT, int ld_du, int ld_A, int ld_d, int ld_x){
	__shared__ T s_P[DIM_P_r*DIM_P_c];
	__shared__ T s_p[DIM_p_r];
	__shared__ T s_AB[DIM_AB_r*DIM_AB_c];
	__shared__ T s_AB2[DIM_AB_r*DIM_AB_c];
	__shared__ T s_H[DIM_H_r*DIM_H_c];
	__shared__ T s_g[DIM_g_r];
	__shared__ T s_K[DIM_K_r*DIM_K_c];
	__shared__ T s_du[DIM_du_r];
	__shared__ T s_dJ[2*DIM_du_r];
	__shared__ T s_Huu[2*DIM_Huu_r*DIM_Huu_c];
	__shared__ T s_dx[STATE_SIZE];
	__shared__ T s_temp[DIM_Huu_r + DIM_Huu_c + 1];
  	for (int block = blockIdx.x; block < M_BLOCKS_B; block += gridDim.x){
	    int ks = (N_BLOCKS_B*(block+1)-1);		int LIN_XFRM_FLAG = LINEAR_TRANSFORM_SWITCH;	int iterCount;
	    T *b_Pprev = DIM_P_c*ld_P*(ks-1) + d_P;		T *b_pprev = ld_p*(ks-1) + d_p;
	    T *b_H = DIM_H_c*ld_H*ks + d_H;				T *b_g = ld_g*ks + d_g;
	    T *b_P, *b_p, *b_AB, *b_KT, *b_du, *b_d, *b_Bdu, *b_ApBK;
	    // be careful that if you are the final block you need to simply copy your Hxx cost -> P and gx -> p back one timestep
	    if (ks == NUM_TIME_STEPS - 1){
	        copyMat<T,DIM_P_r,DIM_P_c>(b_Pprev, b_H, ld_P, ld_H);		copyMat<T,DIM_p_r,DIM_p_c>(b_pprev, b_g, ld_p, ld_g);
	        // then update pointers
	        b_P = b_Pprev;		b_p = b_pprev;	b_Pprev -= DIM_P_c*ld_P;	b_pprev -= ld_p;		ks--;
	        b_H -= DIM_H_c*ld_H;	b_g -= ld_g;	iterCount = N_BLOCKS_B - 2;		LIN_XFRM_FLAG = 0;
	    }
	    // else read first from pp to ensure no weird asynchronous stuff (hard to compare) -- and apply linxfrm if asked
	    else{iterCount = N_BLOCKS_B - 1;	b_P = DIM_P_c*ld_P*ks + (FORCE_PARALLEL ? d_Pp : d_P);	b_p = ld_p*ks + (FORCE_PARALLEL ? d_pp : d_p);}
	    // in either case load the rest of the vars
	    b_AB = DIM_AB_c*ld_AB*ks + d_AB;	b_KT = DIM_KT_c*ld_KT*ks + d_KT;	b_du = ld_du*ks + d_du;
	    b_d = ld_d*ks + d_d;				b_Bdu = ld_d*ks + d_Bdu;			b_ApBK = DIM_A_c*ld_A*ks + d_ApBK;
	    // zero the expected cost reduction
	    zeroSharedMem<T,2*DIM_du_r>(s_dJ);
	    // load P,p and compute linear transform if needed
	    linearXfrmOrLoad<T>(s_P,s_p,s_dx,d_x,d_xp,ks,ld_x,ld_P,LIN_XFRM_FLAG,b_P,b_p);
	    hd__syncthreads();
	    // loop back in time
	    for (int iter = iterCount; iter >= 0; iter--){
	        // BACKPROP CTG store in s_H and s_g uses s_AB, s_AB2, s_P, s_p, b_H, b_g, b_AB, b_P, b_p, b_d, s_dx, rho
	        backprop<T>(s_H,s_g,s_AB,s_AB2,s_P,s_p,b_d,rho,iter,ld_H,ld_AB,ld_P,b_H,b_g,b_AB);
	        hd__syncthreads();
	        // COMPUTE KT and du store in s_K, s_du, b_KT, b_du and use s_H, s_Huu, s_g, d_err, rho to compute
	        if (DIM_Huu_r == 1){ // (in dim1 case just 1/Huu)
	            if(computeKTdu_dim1<T>(s_K,s_du,s_H,s_g,rho,ld_KT,b_KT,b_du)){d_err[block] = 1; return;}
	        }
	        else {
	            if (DIM_Huu_r == 4){if (invHuu_dim4<T>(s_H,s_Huu,rho)){d_err[block] = 1; return;}} // error so return else continue
	            else {if (invHuu<T,1>(s_H,s_Huu,rho,s_temp)){d_err[block] = 1; return;}} // error so return else continue
	            hd__syncthreads();
	            computeKTdu<T>(s_K,s_du,s_H,s_g,&s_Huu[DIM_Huu_r*DIM_Huu_c],ld_KT,b_KT,b_du); // InvHuu stored in second half of variable
	        } 
	        hd__syncthreads();
	        // COMPUTE CTG store in b_Pprev, b_pprev, s_P, s_p, compute from S_H, s_g, s_K, s_du, s_AB2
	        // note: if first timestep can skip
	        if (iter != 0 || blockIdx.x != 0){computeCTG<T>(s_P,s_p,s_H,s_g,s_K,s_du,s_AB2,iter,ld_P,b_Pprev,b_pprev);}
	        // COMPUTE vars for Forward Sweep store in b_ApBK, b_Bdu, compute from s_AB, s_K, s_du
	        // note if M_BLOCKS_F == 1 can skip
	        if (M_BLOCKS_F > 1){computeFSVars<T>(b_ApBK,b_Bdu,s_AB,s_K,s_du,ld_A);}
	        // COMPUTE expected cost reduction store in s_dJ, compute from s_H, s_g, and s_du
	        if (USE_EXP_RED){computeExpRed<T>(s_dJ,s_H,s_g,s_du);}
	        // Then update the pointers for the next pass if we need to have one
	        // We propagate back Pkp1 for each ABk so we store such that we can immediately do the next pass of math so
	        // Store each Pkp1 in each Pk and just leaving Pn as zeros (as the H which is computed at the next pass)
	        if (iter != 0){
		        b_P = b_Pprev;				b_p = b_pprev;				b_AB -= DIM_AB_c*ld_AB;
		        b_H -= DIM_H_c*ld_H;		b_g -= ld_g;				b_KT -= DIM_KT_c*ld_KT;
		        b_du -= ld_du;				b_d -= ld_d;				b_Bdu -= ld_d;				
		        b_ApBK -= DIM_A_c*ld_A;		b_Pprev -= DIM_P_c*ld_P;	b_pprev -= DIM_p_c*ld_p;
	        }
	        __syncthreads();
	    }
	    // note successs and save expected cost
	    // assuming DIM_du_r is small this is faster to just do linearly than some complex parallel reduction
	    if (threadIdx.x == 0 && threadIdx.y == 0){
	        for (int j=1; j<DIM_du_r; j++){s_dJ[0] += s_dJ[j];			 s_dJ[DIM_du_r] += s_dJ[DIM_du_r + j];}
	        d_err[block] = 0;				 d_dJexp[2*block] = s_dJ[0];	 d_dJexp[2*block+1] = s_dJ[DIM_du_r];
	    }
	}
}

template <typename T>
__host__
void backPassThreaded(threadDesc_t desc, T *AB, T *P, T *p, T *Pp, T *pp, T *H, T *g, T *KT, T *du, T *d, T *ApBK, T *Bdu, T *x, T *xp, T *dJexp,\
 					  int *err, int ld_AB, int ld_P, int ld_p, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_A, int ld_d, int ld_x, T rho){
   	T *b_AB, *b_P, *b_p, *b_H, *b_g, *b_KT, *b_du, *b_d, *b_Pprev, *b_pprev, *b_ApBK, *b_Bdu;
   	T s_AB2[DIM_AB_r*DIM_AB_c];		T s_K[DIM_K_r*DIM_K_c];				T s_du[DIM_du_r*DIM_du_c];
   	T s_dx[STATE_SIZE];				T s_Huu[2*DIM_Huu_r*DIM_Huu_c];		T s_temp[DIM_Huu_r + DIM_Huu_c + 1];
   	int i, ks, iterCount, LIN_XFRM_FLAG;
  	// zero the expected cost reduction
  	dJexp[2*desc.tid] = 0;	dJexp[2*desc.tid+1] = 0;
   	for (unsigned int i2=0; i2<desc.reps; i2++){
      	i = (desc.tid+i2*desc.dim);		LIN_XFRM_FLAG = LINEAR_TRANSFORM_SWITCH;	ks = (N_BLOCKS_B*(i+1)-1);
	    b_Pprev = DIM_P_c*ld_P*(ks-1) + P;		b_pprev = ld_p*(ks-1) + p;
	    b_H = DIM_H_c*ld_H*ks + H;				b_g = ld_g*ks + g;
	    // be careful that if you are the final block you need to simply copy your Hxx cost -> P and gx -> p back one timestep
	    if (ks == NUM_TIME_STEPS - 1){
	      copyMat<T,DIM_P_r,DIM_P_c>(b_Pprev, b_H, ld_P, ld_H);		copyMat<T,DIM_p_r,DIM_p_c>(b_pprev, b_g, ld_p, ld_g);
	      // then update pointers
	      b_P = b_Pprev;		b_p = b_pprev;		b_Pprev -= DIM_P_c*ld_P;	b_pprev -= ld_p;		ks--;
	      b_H -= DIM_H_c*ld_H;	b_g -= ld_g;		LIN_XFRM_FLAG = 0;			iterCount = N_BLOCKS_B - 2;
	    }
	    // else read first from pp to ensure no weird asynchronous stuff (hard to compare) -- and apply linxfrm if asked
	    else{iterCount = N_BLOCKS_B - 1;	b_P = DIM_P_c*ld_P*ks + (FORCE_PARALLEL ? Pp : P);	b_p = ld_p*ks + (FORCE_PARALLEL ? pp : p);}
	    // in either case load the rest of the vars
	    b_AB = DIM_AB_c*ld_AB*ks + AB;		b_KT = DIM_KT_c*ld_KT*ks + KT;		b_du = ld_du*ks + du;
	    b_d = ld_d*ks + d;					b_Bdu = ld_d*ks + Bdu;				b_ApBK = DIM_A_c*ld_A*ks + ApBK;
  		// compute linear transform if needed
	    linearXfrmOrLoad<T>(b_P,b_p,s_dx,x,xp,ks,ld_x,ld_P,LIN_XFRM_FLAG);
      	// then loop back by block
      	for (int iter = iterCount; iter >= 0; iter--){
        	// BACKPROP CTG store in s_H and s_g uses s_AB2, s_p, b_H, b_g, b_AB, b_P, b_p, rho
        	backprop<T>(b_H, b_g, b_AB, s_AB2, b_P, b_p, b_d, rho, iter, ld_H, ld_AB, ld_P);
      		// COMPUTE KT and du
			// then we need to compute the inverse of Huu and compute KT and du 
			if (DIM_Huu_r == 1){ // (in dim1 case just 1/Huu) 
				if(computeKTdu_dim1<T>(s_K,s_du,b_H,b_g,rho,ld_KT,b_KT,b_du)){err[desc.tid] = 1; return;}
	      	}	
         	else {
         		if (DIM_Huu_r == 4){if (invHuu_dim4<T>(b_H,s_Huu,rho)){err[desc.tid] = 1; return;}} // error so return else continue
         		else {if (invHuu<T>(b_H,s_Huu,rho,s_temp)){err[desc.tid] = 1; return;}} // error so return else continue
            	computeKTdu<T>(s_K,s_du,b_H,b_g,&s_Huu[DIM_Huu_r*DIM_Huu_c],ld_KT,b_KT,b_du);
         	}
      		// COMPUTE CTG
         	if (iter != 0 || i != 0){computeCTG<T>(b_Pprev,b_pprev,b_H,b_g,s_K,s_du,s_AB2,iter,ld_P);}
      		// COMPUTE vars for Forward Sweep (ApBK and Bdu)
         	if (M_BLOCKS_F > 1){computeFSVars<T>(b_ApBK,b_Bdu,b_AB,s_K,s_du,ld_A);}
      		// COMPUTE expected cost reduction
         	if (USE_EXP_RED){computeExpRed<T>(&dJexp[2*desc.tid],b_H,b_g,s_du);}
			// Then update the pointers for the next pass
			// We propagate back Pkp1 for each ABk so we store such that we can immediately do the next pass of math so
			// Store each Pkp1 in each Pk and just leaving Pn as zeros (as the H which is computed at the next pass)
			b_P = b_Pprev;				b_p = b_pprev;				b_AB -= DIM_AB_c*ld_AB;
			b_H -= DIM_H_c*ld_H;		b_g -= ld_g;				b_KT -= DIM_KT_c*ld_KT;
			b_du -= ld_du;				b_d -= ld_d;				b_Bdu -= ld_d;
			b_ApBK -= DIM_A_c*ld_A;		b_Pprev -= DIM_P_c*ld_P;	b_pprev -= ld_p;
  		}
   		// note successs
      	err[desc.tid] = 0;
   	}
}

// full backward pass returns 0 on success and 1 on maxRho error
template <typename T>
__host__ __forceinline__
int backwardPassGPU(T *d_AB, T *d_P, T *d_p, T *d_Pp, T *d_pp, T *d_H, T *d_g, T *d_KT, T *d_du, T *d_d, T *d_ApBK, T *d_Bdu, \
                    T *d_x, T *d_xp, T *d_dJexp, int *err, int *d_err, T *rho, T *drho, cudaStream_t *streams, dim3 dimms, \
                    int ld_AB, int ld_P, int ld_p, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_A, int ld_d, int ld_x){
  	while(1){
	    int fail = 0;
	    // launch the kernel to compute the back bass in blocks       
	    backPassKern<T><<<M_BLOCKS_B,dimms,0,streams[0]>>>(d_AB,d_P,d_p,d_Pp,d_pp,d_H,d_g,d_KT,d_du,d_d,d_ApBK,d_Bdu,d_x,d_xp,d_dJexp,d_err,
	                                                *rho,ld_AB,ld_P,ld_p,ld_H,ld_g,ld_KT,ld_du,ld_A,ld_d,ld_x);
	    // check for an error
	    gpuErrchk(cudaStreamSynchronize(streams[0]));
	    gpuErrchk(cudaMemcpy(err, d_err, M_BLOCKS_B*sizeof(int), cudaMemcpyDeviceToHost));
	    for (int i = 0; i < M_BLOCKS_B; i++){fail |= err[i];}
	    // if an error then reset and increase rho
	    if (fail){
	      	*drho = max((*drho)*static_cast<T>(RHO_FACTOR),static_cast<T>(RHO_FACTOR));
	      	*rho = min((*rho)*(*drho), static_cast<T>(RHO_MAX));
	      	if (*rho == static_cast<T>(RHO_MAX) && !IGNORE_MAX_ROX_EXIT){return 1;}
	      	else { // try to do the factorization again with a larger rho
		        if (DEBUG_SWITCH){printf("[!]Inversion Failed Increasing Rho\n");}
		        // need to reset the d_P, d_p
		        gpuErrchk(cudaMemcpyAsync(d_P,d_Pp,ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[0]));
		        gpuErrchk(cudaMemcpyAsync(d_p,d_pp,ld_p*DIM_p_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[1]));
		        gpuErrchk(cudaStreamSynchronize(streams[0]));
		        gpuErrchk(cudaStreamSynchronize(streams[1]));
		        continue;
	      	}
	    }
	    // was a success so break
	    else{break;}
  	}
		return 0;
}

// full backward pass returns 0 on success and 1 on maxRho error
template <typename T>
__host__ __forceinline__
int backwardPassCPU(T *AB, T *P, T *p, T *Pp, T *pp, T *H, T *g, T *KT, T *du, T *d, T *dp,
					T *ApBK, T *Bdu, T *x, T *xp, T *dJexp, int *err, T *rho, T *drho, 
					std::thread *threads, int ld_AB, int ld_P, int ld_p, int ld_H, int ld_g, 
					int ld_KT, int ld_du, int ld_A, int ld_d, int ld_x){
	while(1){
		int fail = 0;
     	// compute the back pass threaded
     	threadDesc_t desc;    desc.dim = BP_THREADS;
     	for (unsigned int thread_i = 0; thread_i < BP_THREADS; thread_i++){
        	desc.tid = thread_i;   desc.reps = compute_reps(thread_i,BP_THREADS,M_BLOCKS_B);
        	threads[thread_i] = std::thread(&backPassThreaded<T>, desc, std::ref(AB), std::ref(P), std::ref(p), std::ref(Pp), std::ref(pp), std::ref(H), 
        														  std::ref(g), std::ref(KT), std::ref(du), std::ref(d), std::ref(ApBK), std::ref(Bdu), 
        														  std::ref(x), std::ref(xp), std::ref(dJexp), std::ref(err), ld_AB, ld_P, ld_p, ld_H, 
        														  ld_g, ld_KT, ld_du, ld_A, ld_d, ld_x, *rho);
        	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, thread_i);}
     	}
     	// while this runs save d -> dp if M_BLOCKS_F > 1
     	if (M_BLOCKS_F > 1){
     		threads[BP_THREADS] = std::thread(memcpy, std::ref(dp), std::ref(d), ld_d*NUM_TIME_STEPS*sizeof(T));
     		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, BP_THREADS);}
 	 	}
     	for (unsigned int thread_i = 0; thread_i < BP_THREADS; thread_i++){threads[thread_i].join(); fail |= err[thread_i];}
 		if (M_BLOCKS_F > 1){threads[BP_THREADS].join();}
     	// if an error then reset and increase rho
	    if (fail){
	      	*drho = max((*drho)*static_cast<T>(RHO_FACTOR),static_cast<T>(RHO_FACTOR));
	      	*rho = min((*rho)*(*drho), static_cast<T>(RHO_MAX));
	      	if (*rho == static_cast<T>(RHO_MAX) && !IGNORE_MAX_ROX_EXIT){return 1;}
	      	else { // try to do the factorization again with a larger rho
	        	if (DEBUG_SWITCH){printf("[!]Inversion Failed Increasing Rho\n");}
	        	// need to reset the d_P, d_p
	        	threads[0] = std::thread(memcpy, std::ref(P), std::ref(Pp), ld_P*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
	        	if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
           		threads[1] = std::thread(memcpy, std::ref(p), std::ref(pp), ld_p*NUM_TIME_STEPS*sizeof(T));
           		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
               	threads[0].join();
               	threads[1].join();
	        	continue;
	      	}
	    }
    	// was a success so break
    	else{break;}
		}
		return 0;
	}
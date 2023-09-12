/*****************************************************************
 * DDP Algorithm Wrappers
 * (currently only supports iLQR)
 *
 *  runiLQR_(C/G)PU(2) note: 2 is for parallel line search for CPU
 *****************************************************************/

template <typename T>
__host__ __forceinline__
void runiLQR_GPU(T *x0, T *u0, T *KT0, T *P0, T *p0, T *d0, T *xGoal, T *Jout, int *alphaOut, int forwardRolloutFlag, int clearVarsFlag, int ignoreFirstDefectFlag,
				 double *tTime, double *simTime, double *sweepTime, double *bpTime, double *nisTime, double *initTime, cudaStream_t *streams,
				 T **d_x, T **h_d_x, T *d_xp, T *d_xp2, T **d_u, T **h_d_u, T *d_up,
				 T *d_P, T *d_p, T *d_Pp, T *d_pp, T *d_AB, T *d_H, T *d_g, T *d_KT, T *d_du,
				 T **d_d, T **h_d_d, T *d_dp, T *d_dT, T *d, T *d_ApBK, T *d_Bdu, T *d_dM,
				 T *alpha, T *d_alpha, int *alphaIndex, T *d_JT, T *J, T *dJexp, T *d_dJexp, T *d_xGoal,
				 int *err, int *d_err, int ld_x, int ld_u, int ld_P, int ld_p, int ld_AB, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_d, int ld_A, 
				 T *d_I = nullptr, T *d_Tbody = nullptr,
				 T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, \
   			     T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, \
   			     T R_EE = _R_EE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, \
   			     T Q1 = _Q1, T Q2 = _Q2, T R = _R, T QF1 = _QF1, T QF2 = _QF2){
	// INITIALIZE THE ALGORITHM	//
	struct timeval start, end, start2, end2;	gettimeofday(&start,NULL);	gettimeofday(&start2,NULL);
	T prevJ, dJ, z; 	int iter = 1;	T rho = RHO_INIT; 	T drho = 1.0;	 *alphaIndex = 0;

	// define kernel dimms
	dim3 ADimms(DIM_A_r,1);//DIM_A_r,DIM_A_c);
	dim3 bpDimms(8,7); 				dim3 dynDimms(8,7);//(36,7);
	dim3 FPBlocks(M_BLOCKS_F,NUM_ALPHA);	dim3 gradBlocks(DIM_AB_c,NUM_TIME_STEPS-1);		dim3 intDimms(NUM_TIME_STEPS-1,1);
	if(USE_FINITE_DIFF){intDimms.y = STATE_SIZE_PDDP + CONTROL_SIZE;}

	// load and clear variables as requested and init the alg
	loadVarsGPU<T>(d_x,h_d_x,d_xp,x0,d_u,h_d_u,d_up,u0,d_P,d_Pp,P0,d_p,d_pp,p0,d_KT,KT0,d_du,d_dT,d_d,h_d_d,d0,d_AB,d_err,xGoal,d_xGoal,d_alpha,
				   d_Tbody,d_I,d_JT,clearVarsFlag,forwardRolloutFlag,streams,dynDimms,ld_x,ld_u,ld_P,ld_p,ld_KT,ld_du,ld_d,ld_AB,
				   Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
	initAlgGPU<T>(d_x,h_d_x,d_xp,d_xp2,d_u,h_d_u,d_up,d_d,h_d_d,d_dp,d_dT,d_AB,d_H,d_g,d_KT,d_du,d_JT,&prevJ,d_xGoal,d_alpha,alphaIndex,
			      alphaOut,Jout,streams,dynDimms,intDimms,forwardRolloutFlag,ld_x,ld_u,ld_d,ld_AB,ld_H,ld_g,ld_KT,ld_du,d_I,d_Tbody,
			      Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
	gettimeofday(&end2,NULL);
	*initTime = time_delta_ms(start2,end2);
	// INITIALIZE THE ALGORITHM //

	// debug print
	if (DEBUG_SWITCH){
		gpuErrchk(cudaMemcpy(&prevJ, &d_JT[*alphaIndex], sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(x0, h_d_x[*alphaIndex], ld_x*DIM_x_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost));
		printf("Iter[0] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] Rho[%f]\n",
			x0[ld_x*DIM_x_c*(NUM_TIME_STEPS-1)],x0[ld_x*DIM_x_c*(NUM_TIME_STEPS-1)+1],prevJ,*alphaIndex,rho);
	}

	// now start computing iterates
	while(1){
		// BACKWARD PASS //
			gettimeofday(&start2,NULL);
			// run full backward pass if it fails we have maxed our regularizer and need to exit
			if (backwardPassGPU<T>(d_AB,d_P,d_p,d_Pp,d_pp,d_H,d_g,d_KT,d_du,h_d_d[*alphaIndex],d_ApBK,d_Bdu,
								   h_d_x[*alphaIndex],d_xp2,d_dJexp,err,d_err,&rho,&drho,streams,bpDimms,
								   ld_AB,ld_P,ld_p,ld_H,ld_g,ld_KT,ld_du,ld_A,ld_d,ld_x)){
				if (DEBUG_SWITCH){printf("Exiting for maxRho\n");}
				break;
			}
			// make sure everything that was supposed to finish did by now (incuding previous NIS stuff)
			gpuErrchk(cudaDeviceSynchronize());
			gettimeofday(&end2,NULL);
			bpTime[iter-1] = time_delta_ms(start2,end2);
		// BACKWARD PASS //

		// FORWARD PASS //
			// FORWARD SWEEP //
				gettimeofday(&start2,NULL);
				// Sweep forward with all alpha in parallel if applicable
				if (M_BLOCKS_F > 1){
					forwardSweepKern<T><<<NUM_ALPHA,ADimms,0,streams[0]>>>(d_x,d_ApBK,d_Bdu,h_d_d[*alphaIndex],d_xp,d_alpha,ld_x,ld_d,ld_A);
					gpuErrchk(cudaPeekAtLastError());	gpuErrchk(cudaDeviceSynchronize());
				}
				gettimeofday(&end2,NULL);
				sweepTime[iter-1] = time_delta_ms(start2,end2);
			// FORWARD SWEEP //

			// FORWARD SIM //
				gettimeofday(&start2,NULL);
				// Simulate forward with all alpha in parallel with MS, compute costs and line search
				forwardSimGPU<T>(d_x,d_xp,d_xp2,d_u,d_KT,d_du,alpha,d_alpha,d,d_d,d_dT,dJexp,d_dJexp,J,d_JT,d_xGoal,&dJ,&z,prevJ,
							     streams,dynDimms,FPBlocks,alphaIndex,&ignoreFirstDefectFlag,ld_x,ld_u,ld_KT,ld_du,ld_d,d_I,d_Tbody,
							     Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
				gettimeofday(&end2,NULL);
				simTime[iter-1] = time_delta_ms(start2,end2);
			// FORWARD SIM //
		// FORWARD PASS //
				
		// NEXT ITERATION SETUP //
			gettimeofday(&start2,NULL);
			// process accept or reject of traj and test for exit
			if (acceptRejectTrajGPU<T>(h_d_x,d_xp,h_d_u,d_up,h_d_d,d_dp,J,&prevJ,&dJ,&rho,&drho,alphaIndex,alphaOut,Jout,&iter,streams,ld_x,ld_u,ld_d)){
				gettimeofday(&end2,NULL);
				nisTime[iter-1] = time_delta_ms(start2,end2);
				break;
			}

			// if we have gotten here then prep for next pass
			nextIterationSetupGPU<T>(d_x,h_d_x,d_xp,d_u,h_d_u,d_up,d_d,h_d_d,d_dp,d_AB,d_H,d_g,d_P,d_p,d_Pp,d_pp,d_xGoal,alphaIndex,
								     streams,dynDimms,intDimms,ld_x,ld_u,ld_d,ld_AB,ld_H,ld_g,ld_P,ld_p,d_I,d_Tbody,
								     Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
			gettimeofday(&end2,NULL);
			nisTime[iter-2] = time_delta_ms(start2,end2);
		// NEXT ITERATION SETUP //

		// debug print
		if (DEBUG_SWITCH){
			gpuErrchk(cudaMemcpy(x0, h_d_x[*alphaIndex], STATE_SIZE_PDDP*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost));
			printf("Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
					iter-1,x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,*alphaIndex,rho,dJ,z,d[*alphaIndex]);
		}
	}

	// EXIT Handling
		// on exit make sure everything finishes
		gpuErrchk(cudaDeviceSynchronize());
		if (DEBUG_SWITCH){
			gpuErrchk(cudaMemcpy(x0, h_d_x[*alphaIndex], STATE_SIZE_PDDP*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost));
			printf("Exit with Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
				iter,x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,*alphaIndex,rho,dJ,z,d[*alphaIndex]);
		}

		// Bring back the final state and control (and compute and bring back the defect)
		gettimeofday(&start2,NULL);
		storeVarsGPU(h_d_x,x0,h_d_u,u0,alphaIndex,streams,ld_x,ld_u,d_d,d_dT,d,ld_d);
		gettimeofday(&end2,NULL);
		gettimeofday(&end,NULL);
		*initTime += time_delta_ms(start2,end2);
		*tTime = time_delta_ms(start,end);

		// print the result
		printf("GPU Parallel blocks:[%d] t:[%f] with FP[%f], FS[%f], BP[%f], NIU[%f] Xf:[%.4f, %.4f] iters:[%d] cost:[%f] max_d[%f]\n",
					M_BLOCKS_B,*tTime,*simTime,*sweepTime,*bpTime,*nisTime,x0[ld_x*(NUM_TIME_STEPS-1)],x0[ld_x*(NUM_TIME_STEPS-1)+1],iter,prevJ,d[*alphaIndex]);
		if (DEBUG_SWITCH){printf("\n");}
	// EXIT Handling
}
	
template <typename T>
__host__ __forceinline__
void runiLQR_CPU(T *x0, T *u0, T *KT0, T *P0, T *p0, T *d0, T *xGoal, T *Jout, int *alphaOut, int forwardRolloutFlag, int clearVarsFlag, int ignoreFirstDefectFlag,
				 double *tTime, double *simTime, double *sweepTime, double *bpTime, double *nisTime, double *initTime,
				 T *x, T *xp, T *xp2, T *u, T *up, T *P, T *p, T *Pp, T *pp, 
				 T *AB, T *H, T *g, T *KT, T *du, T *d, T *dp, 
				 T *ApBK, T *Bdu, T *alpha, T *JT, T *dJexp,  int *err,
				 int ld_x, int ld_u, int ld_P, int ld_p, int ld_AB, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_d, int ld_A,
				 T *I = nullptr, T *Tbody = nullptr,
				 T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, \
	   			 T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, \
	   			 T R_EE = _R_EE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, \
	   			 T Q1 = _Q1, T Q2 = _Q2, T R = _R, T QF1 = _QF1, T QF2 = _QF2){
	// INITIALIZE THE ALGORITHM //
		struct timeval start, end, start2, end2;	gettimeofday(&start,NULL);	gettimeofday(&start2,NULL);
		T prevJ, dJ, J, z, maxd = 0;	int iter = 1;
		T rho = RHO_INIT;	T drho = 1.0;	int alphaIndex = 0;

		// define array for general threads
		std::thread threads[MAX_CPU_THREADS];
		
		// load in vars and init the alg
		loadVarsCPU<T>(x,xp,x0,u,up,u0,P,Pp,P0,p,pp,p0,KT,KT0,du,d,d0,AB,err,clearVarsFlag,forwardRolloutFlag,
					   alpha,I,Tbody,xGoal,JT,threads,ld_x,ld_u,ld_P,ld_p,ld_KT,ld_du,ld_d,ld_AB,
					   Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
		initAlgCPU<T>(x,xp,xp2,u,up,AB,H,g,KT,du,d,JT,Jout,&prevJ,alpha,alphaOut,xGoal,threads,
		               forwardRolloutFlag,ld_x,ld_u,ld_AB,ld_H,ld_g,ld_KT,ld_du,ld_d,I,Tbody,
		               Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
		gettimeofday(&end2,NULL);
		*initTime = time_delta_ms(start2,end2);
	// INITIALIZE THE ALGORITHM //

	// debug print -- so ready to start
	if (DEBUG_SWITCH){
		printf("Iter[0] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] Rho[%f]\n",
					x[ld_x*(NUM_TIME_STEPS-1)],x[ld_x*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho);
	}

	while(1){
		
		// BACKWARD PASS //
			gettimeofday(&start2,NULL);
			backwardPassCPU<T>(AB,P,p,Pp,pp,H,g,KT,du,d,dp,ApBK,Bdu,x,xp2,dJexp,err,&rho,&drho,&threads[0],ld_AB,ld_P,ld_p,ld_H,ld_g,ld_KT,ld_du,ld_A,ld_d,ld_x);
			gettimeofday(&end2,NULL);
			bpTime[iter-1] = time_delta_ms(start2,end2);
		// BACKWARD PASS //

		// FORWARD PASS //
			dJ = -1.0;	alphaIndex = 0;	sweepTime[iter-1] = 0.0;	simTime[iter-1] = 0.0;
			while(1){
				// FORWARD SWEEP //
					gettimeofday(&start2,NULL);
					// Do the forward sweep if applicable
					if (M_BLOCKS_F > 1){forwardSweep<T>(x, ApBK, Bdu, d, xp, alpha[alphaIndex], ld_x, ld_d, ld_A);}
					gettimeofday(&end2,NULL);
					sweepTime[iter-1] += time_delta_ms(start2,end2);
				// FORWARD SWEEP //

				// FORWARD SIM //
					gettimeofday(&start2,NULL);
					int err = forwardSimCPU<T>(x,xp,xp2,u,up,KT,du,d,dp,dJexp,JT,alpha[alphaIndex],xGoal,
		    								   &J,&dJ,&z,prevJ,&ignoreFirstDefectFlag,&maxd,threads,ld_x,ld_u,ld_KT,ld_du,ld_d,I,Tbody,
		    								   Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
					gettimeofday(&end2,NULL);	
					simTime[iter-1] += time_delta_ms(start2,end2);
					if(err){if (alphaIndex < NUM_ALPHA - 1){alphaIndex++; continue;} else{alphaIndex = -1; break;}} else{break;}
				// FORWARD SIM //
			}
		// FORWARD PASS //

		// NEXT ITERATION SETUP //
			gettimeofday(&start2,NULL);    
			// process accept or reject of traj and test for exit
			if (acceptRejectTrajCPU<T>(x,xp,u,up,d,dp,J,&prevJ,&dJ,&rho,&drho,&alphaIndex,alphaOut,Jout,&iter,threads,ld_x,ld_u,ld_d)){
				gettimeofday(&end2,NULL);
				nisTime[iter-1] = time_delta_ms(start2,end2);
				break;
			}
			// if we have gotten here then prep for next pass
			nextIterationSetupCPU<T>(x,xp,u,up,d,dp,AB,H,g,P,p,Pp,pp,xGoal,threads,ld_x,ld_u,ld_d,ld_AB,ld_H,ld_g,ld_P,ld_p,I,Tbody,
									 Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
			gettimeofday(&end2,NULL);
			nisTime[iter-2] = time_delta_ms(start2,end2);
		// NEXT ITERATION SETUP //

		if (DEBUG_SWITCH){
			printf("Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
						iter-1,x[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],x[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho,dJ,z,maxd);
		}
	}

	// EXIT Handling
		if (DEBUG_SWITCH){
			printf("Exit with Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
						iter,x[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],x[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho,dJ,z,maxd);
		}
		// Bring back the final state and control (and compute final d if needed)
		gettimeofday(&start2,NULL);
		storeVarsCPU(x,x0,u,u0,threads,ld_x,ld_u,d,&maxd,ld_d);
		gettimeofday(&end2,NULL);
		gettimeofday(&end,NULL);
		*initTime += time_delta_ms(start2,end2);
		*tTime = time_delta_ms(start,end);

		printf("CPU Parallel blocks:[%d] t:[%f] with FP[%f], FS[%f], BP[%f], NIU[%f] Xf:[%.4f, %.4f] iters:[%d] cost:[%f] max_d[%f]\n",
					M_BLOCKS_B,*tTime,*simTime,*sweepTime,*bpTime,*nisTime,x0[ld_x*(NUM_TIME_STEPS-1)],x0[ld_x*(NUM_TIME_STEPS-1)+1],iter,prevJ,maxd);
		if (DEBUG_SWITCH){printf("\n");}
	// EXIT Handling
}

template <typename T>
__host__ __forceinline__
void runiLQR_CPU2(T *x0, T *u0, T *KT0, T *P0, T *p0, T *d0, T *xGoal, T *Jout, int *alphaOut, 
				 int forwardRolloutFlag, int clearVarsFlag, int ignoreFirstDefectFlag,
				 double *tTime, double *simTime, double *sweepTime, double *bpTime, double *nisTime, double *initTime,
				 T **xs, T *x, T *xp, T *xp2, T **us, T *u, T *up, T *P, T *p, T *Pp, T *pp, 
				 T *AB, T *H, T *g, T *KT, T *du, T **ds, T *d, T *dp, 
				 T *ApBK, T *Bdu, T *alphas, T **JTs, T *dJexp,  int *err,
				 int ld_x, int ld_u, int ld_P, int ld_p, int ld_AB, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_d, int ld_A,
				 T *I = nullptr, T *Tbody = nullptr,
				 T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, \
	   			 T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, \
	   			 T R_EE = _R_EE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, \
	   			 T Q1 = _Q1, T Q2 = _Q2, T R = _R, T QF1 = _QF1, T QF2 = _QF2){
	// INITIALIZE THE ALGORITHM //
		struct timeval start, end, start2, end2;	gettimeofday(&start,NULL);	gettimeofday(&start2,NULL);
		T prevJ, dJ, J, z, maxd = 0;	int iter = 1;
		T rho = RHO_INIT;	T drho = 1.0;	int alphaIndex = 0;

		// define array for general threads
		std::thread threads[MAX_CPU_THREADS];
		
		// load in vars and init the alg
		loadVarsCPU<T>(x,xp,x0,u,up,u0,P,Pp,P0,p,pp,p0,KT,KT0,du,d,d0,AB,err,clearVarsFlag,forwardRolloutFlag,
					   alphas,I,Tbody,xGoal,JTs[0],threads,ld_x,ld_u,ld_P,ld_p,ld_KT,ld_du,ld_d,ld_AB,
					   Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
		initAlgCPU2<T>(xs,x,xp,xp2,us,u,up,AB,H,g,KT,du,ds,d,JTs[0],Jout,&prevJ,alphas,alphaOut,xGoal,threads,
		               forwardRolloutFlag,ld_x,ld_u,ld_AB,ld_H,ld_g,ld_KT,ld_du,ld_d,I,Tbody,
		               Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
		gettimeofday(&end2,NULL);
		*initTime = time_delta_ms(start2,end2);
	// INITIALIZE THE ALGORITHM //

	// debug print -- so ready to start
	if (DEBUG_SWITCH){
		printf("Iter[0] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] Rho[%f]\n",
					xs[alphaIndex][ld_x*(NUM_TIME_STEPS-1)],xs[alphaIndex][ld_x*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho);
	}
	while(1){
		// BACKWARD PASS //
			gettimeofday(&start2,NULL);
			backwardPassCPU<T>(AB,P,p,Pp,pp,H,g,KT,du,ds[alphaIndex],dp,ApBK,Bdu,xs[alphaIndex],xp2,dJexp,err,&rho,&drho,threads,ld_AB,ld_P,ld_p,ld_H,ld_g,ld_KT,ld_du,ld_A,ld_d,ld_x);
			gettimeofday(&end2,NULL);
			bpTime[iter-1] = time_delta_ms(start2,end2);
		// BACKWARD PASS //

		// FORWARD PASS //
			dJ = -1.0;	alphaIndex = 0;	sweepTime[iter-1] = 0.0;	simTime[iter-1] = 0.0;
			while(1){
				// FORWARD SWEEP //
					gettimeofday(&start2,NULL);
					// Do the forward sweep if applicable
					if (M_BLOCKS_F > 1){forwardSweep2<T>(xs, ApBK, Bdu, ds, xp, alphas, alphaIndex, threads, ld_x, ld_d, ld_A);}
					gettimeofday(&end2,NULL);
					sweepTime[iter-1] += time_delta_ms(start2,end2);
				// FORWARD SWEEP //

				// FORWARD SIM //
					gettimeofday(&start2,NULL);
					int alphaIndexOut = forwardSimCPU2<T>(xs,xp,xp2,us,up,KT,du,ds,dp,dJexp,JTs,alphas,alphaIndex,xGoal,
		    								              &J,&dJ,&z,prevJ,&ignoreFirstDefectFlag,&maxd,threads,
		    								    		  ld_x,ld_u,ld_KT,ld_du,ld_d,I,Tbody,
		    								    		  Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,
		    								    		  Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
					gettimeofday(&end2,NULL);	
					simTime[iter-1] += time_delta_ms(start2,end2);
					if(alphaIndexOut == -1){ // failed
						if (alphaIndex < NUM_ALPHA - FSIM_ALPHA_THREADS){alphaIndex += FSIM_ALPHA_THREADS; continue;} // keep searching
						else{alphaIndex = -1; break;} // note failure
					} 
					else{alphaIndex = alphaIndexOut; break;} // save success
				// FORWARD SIM //
			}
		// FORWARD PASS //

		// NEXT ITERATION SETUP //
			gettimeofday(&start2,NULL);    
			// process accept or reject of traj and test for exit
			if (acceptRejectTrajCPU2<T>(xs,xp,us,up,ds,dp,J,&prevJ,&dJ,&rho,&drho,&alphaIndex,alphaOut,Jout,&iter,threads,ld_x,ld_u,ld_d)){
				gettimeofday(&end2,NULL);
				nisTime[iter-1] = time_delta_ms(start2,end2);
				if (alphaIndex == -1){alphaIndex = 0;} break;
			}
			// if we have gotten here then prep for next pass
			nextIterationSetupCPU2<T>(xs,xp,us,up,ds,dp,AB,H,g,P,p,Pp,pp,xGoal,threads,&alphaIndex,ld_x,ld_u,ld_d,ld_AB,ld_H,ld_g,ld_P,ld_p,I,Tbody,
									  Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
			gettimeofday(&end2,NULL);
			nisTime[iter-2] = time_delta_ms(start2,end2);
		// NEXT ITERATION SETUP //

		if (DEBUG_SWITCH){
			printf("Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
						iter-1,xs[alphaIndex][DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],xs[alphaIndex][DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho,dJ,z,maxd);
		}
	}

	// EXIT Handling
		if (DEBUG_SWITCH){
			printf("Exit with Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
						iter,xs[alphaIndex][DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],xs[alphaIndex][DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho,dJ,z,maxd);
		}
		// Bring back the final state and control (and compute final d if needed)
		gettimeofday(&start2,NULL);
		storeVarsCPU(xs[alphaIndex],x0,us[alphaIndex],u0,threads,ld_x,ld_u,ds[alphaIndex],&maxd,ld_d);
		gettimeofday(&end2,NULL);
		gettimeofday(&end,NULL);
		*initTime += time_delta_ms(start2,end2);
		*tTime = time_delta_ms(start,end);

		printf("CPU Parallel blocks:[%d] t:[%f] with FP[%f], FS[%f], BP[%f], NIU[%f] Xf:[%.4f, %.4f] iters:[%d] cost:[%f] max_d[%f]\n",
					M_BLOCKS_B,*tTime,*simTime,*sweepTime,*bpTime,*nisTime,x0[ld_x*(NUM_TIME_STEPS-1)],x0[ld_x*(NUM_TIME_STEPS-1)+1],iter,prevJ,maxd);
		if (DEBUG_SWITCH){printf("\n");}
	// EXIT Handling
}

template <typename T>
__host__ __forceinline__
void runSLQ_GPU(T *x0, T *u0, T *KT0, T *P0, T *p0, T *d0, T *xGoal, T *Jout, int *alphaOut, int forwardRolloutFlag, int clearVarsFlag, int ignoreFirstDefectFlag,
				 double *tTime, double *simTime, double *sweepTime, double *bpTime, double *nisTime, double *initTime, cudaStream_t *streams,
				 T **d_x, T **h_d_x, T *d_xp, T *d_xp2, T **d_u, T **h_d_u, T *d_up,
				 T *d_P, T *d_p, T *d_Pp, T *d_pp, T *d_AB, T *d_H, T *d_g, T *d_KT, T *d_du,
				 T **d_d, T **h_d_d, T *d_dp, T *d_dT, T *d, T *d_ApBK, T *d_Bdu, T *d_dM,
				 T *alpha, T *d_alpha, int *alphaIndex, T *d_JT, T *J, T *dJexp, T *d_dJexp, T *d_xGoal,
				 int *err, int *d_err, int ld_x, int ld_u, int ld_P, int ld_p, int ld_AB, int ld_H, int ld_g, int ld_KT, int ld_du, int ld_d, int ld_A, 
				 T *d_I = nullptr, T *d_Tbody = nullptr,
				 T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, \
	   			 T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, \
	   			 T R_EE = _R_EE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, \
	   			 T Q1 = _Q1, T Q2 = _Q2, T R = _R, T QF1 = _QF1, T QF2 = _QF2){
	// INITIALIZE THE ALGORITHM	//
	struct timeval start, end, start2, end2;	gettimeofday(&start,NULL);	gettimeofday(&start2,NULL);
	T prevJ, dJ, z; 	int iter = 1;	T rho = RHO_INIT; 	T drho = 1.0;	 *alphaIndex = 0;

	// define kernel dimms
	dim3 ADimms(DIM_A_r,1);//DIM_A_r,DIM_A_c);
	dim3 bpDimms(8,7); 				dim3 dynDimms(8,7);//(36,7);
	dim3 FPBlocks(M_BLOCKS_F,NUM_ALPHA);	dim3 gradBlocks(DIM_AB_c,NUM_TIME_STEPS-1);		dim3 intDimms(NUM_TIME_STEPS-1,1);
	if(USE_FINITE_DIFF){intDimms.y = STATE_SIZE_PDDP + CONTROL_SIZE;}

	// load and clear variables as requested and init the alg
	loadVarsGPU<T>(d_x,h_d_x,d_xp,x0,d_u,h_d_u,d_up,u0,d_P,d_Pp,P0,d_p,d_pp,p0,d_KT,KT0,d_du,d_dT,d_d,h_d_d,d0,d_AB,d_err,xGoal,d_xGoal,d_alpha,
				   d_Tbody,d_I,d_JT,clearVarsFlag,forwardRolloutFlag,streams,dynDimms,ld_x,ld_u,ld_P,ld_p,ld_KT,ld_du,ld_d,ld_AB,
				   Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
	initAlgGPU<T>(d_x,h_d_x,d_xp,d_xp2,d_u,h_d_u,d_up,d_d,h_d_d,d_dp,d_dT,d_AB,d_H,d_g,d_KT,d_du,d_JT,&prevJ,d_xGoal,d_alpha,alphaIndex,
			      alphaOut,Jout,streams,dynDimms,intDimms,forwardRolloutFlag,ld_x,ld_u,ld_d,ld_AB,ld_H,ld_g,ld_KT,ld_du,d_I,d_Tbody,
			      Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
	gettimeofday(&end2,NULL);
	*initTime = time_delta_ms(start2,end2);
	// INITIALIZE THE ALGORITHM //

	// debug print
	if (DEBUG_SWITCH){
		gpuErrchk(cudaMemcpy(&prevJ, &d_JT[*alphaIndex], sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(x0, h_d_x[*alphaIndex], ld_x*DIM_x_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost));
		printf("Iter[0] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] Rho[%f]\n",
			x0[ld_x*DIM_x_c*(NUM_TIME_STEPS-1)],x0[ld_x*DIM_x_c*(NUM_TIME_STEPS-1)+1],prevJ,*alphaIndex,rho);
	}

	// now start computing iterates
	while(1){
		// BACKWARD PASS //
			gettimeofday(&start2,NULL);
			// run full backward pass if it fails we have maxed our regularizer and need to exit
			if (backwardPassGPU<T>(d_AB,d_P,d_p,d_Pp,d_pp,d_H,d_g,d_KT,d_du,h_d_d[*alphaIndex],d_ApBK,d_Bdu,
								   h_d_x[*alphaIndex],d_xp2,d_dJexp,err,d_err,&rho,&drho,streams,bpDimms,
								   ld_AB,ld_P,ld_p,ld_H,ld_g,ld_KT,ld_du,ld_A,ld_d,ld_x)){
				if (DEBUG_SWITCH){printf("Exiting for maxRho\n");}
				break;
			}
			// make sure everything that was supposed to finish did by now (incuding previous NIS stuff)
			gpuErrchk(cudaDeviceSynchronize());
			gettimeofday(&end2,NULL);
			bpTime[iter-1] = time_delta_ms(start2,end2);
		// BACKWARD PASS //

		// FORWARD PASS //
			
			gettimeofday(&start2,NULL);
			forwardPassSLQGPU(d_x,d_xp,d_xp2,d_u,d_ApBK,d_Bdu,d_du,d_KT,alpha,d_alpha,dJexp,d_dJexp,J,d_JT,
				              d_xGoal,&dJ,&z,prevJ,streams,ADimms,alphaIndex,ld_x,ld_u,ld_d,ld_A,ld_du,ld_KT,
				              Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
			gettimeofday(&end2,NULL);
			sweepTime[iter-1] = time_delta_ms(start2,end2);		simTime[iter-1] = 0;
		// FORWARD PASS //

		// NEXT ITERATION SETUP //
			gettimeofday(&start2,NULL);
			// process accept or reject of traj and test for exit
			if (acceptRejectTrajGPU<T>(h_d_x,d_xp,h_d_u,d_up,h_d_d,d_dp,J,&prevJ,&dJ,&rho,&drho,alphaIndex,alphaOut,Jout,&iter,streams,ld_x,ld_u,ld_d)){
				gettimeofday(&end2,NULL);
				nisTime[iter-1] = time_delta_ms(start2,end2);
				break;
			}

			// if we have gotten here then prep for next pass
			nextIterationSetupGPU<T>(d_x,h_d_x,d_xp,d_u,h_d_u,d_up,d_d,h_d_d,d_dp,d_AB,d_H,d_g,d_P,d_p,d_Pp,d_pp,d_xGoal,alphaIndex,
								     streams,dynDimms,intDimms,ld_x,ld_u,ld_d,ld_AB,ld_H,ld_g,ld_P,ld_p,d_I,d_Tbody,
								     Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,Q1,Q2,R,QF1,QF2);
			gettimeofday(&end2,NULL);
			nisTime[iter-2] = time_delta_ms(start2,end2);
		// NEXT ITERATION SETUP //

		// debug print
		if (DEBUG_SWITCH){
			gpuErrchk(cudaMemcpy(x0, h_d_x[*alphaIndex], STATE_SIZE_PDDP*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost));
			printf("Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
					iter-1,x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,*alphaIndex,rho,dJ,z,d[*alphaIndex]);
		}
	}

	// EXIT Handling
		// on exit make sure everything finishes
		gpuErrchk(cudaDeviceSynchronize());
		if (DEBUG_SWITCH){
			gpuErrchk(cudaMemcpy(x0, h_d_x[*alphaIndex], STATE_SIZE_PDDP*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost));
			printf("Exit with Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
				iter,x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)],x0[DIM_x_c*ld_x*(NUM_TIME_STEPS-1)+1],prevJ,*alphaIndex,rho,dJ,z,d[*alphaIndex]);
		}

		// Bring back the final state and control (and compute and bring back the defect)
		gettimeofday(&start2,NULL);
		storeVarsGPU(h_d_x,x0,h_d_u,u0,alphaIndex,streams,ld_x,ld_u,d_d,d_dT,d,ld_d);
		gettimeofday(&end2,NULL);
		gettimeofday(&end,NULL);
		*initTime += time_delta_ms(start2,end2);
		*tTime = time_delta_ms(start,end);

		// print the result
		printf("GPU Parallel blocks:[%d] t:[%f] with FP[%f], FS[%f], BP[%f], NIU[%f] Xf:[%.4f, %.4f] iters:[%d] cost:[%f] max_d[%f]\n",
					M_BLOCKS_B,*tTime,*simTime,*sweepTime,*bpTime,*nisTime,x0[ld_x*(NUM_TIME_STEPS-1)],x0[ld_x*(NUM_TIME_STEPS-1)+1],iter,prevJ,d[*alphaIndex]);
		if (DEBUG_SWITCH){printf("\n");}
	// EXIT Handling
}

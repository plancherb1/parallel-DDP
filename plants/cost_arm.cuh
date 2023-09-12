/*****************************************************************
 * Kuka Arm Cost Funcs
 *
 * TBD NEED TO ADD DOC HERE
 *****************************************************************/

#if USE_EE_VEL_COST
	#warning "Also the Hessian comp using even EE_VEL xyz is bad so don't use EE_VEL_COST yet\n"
#endif

// Also define limits on torque, pos, velocity and Q/Rs for those
#define SAFETY_FACTOR_P 0.8
#define SAFETY_FACTOR_V 0.8
#define SAFETY_FACTOR_T 0.8
// From URDF
#define TORQUE_LIMIT  (300.0 * SAFETY_FACTOR_T)
#define POS_LIMIT_024 (2.96705972839 * SAFETY_FACTOR_P)
#define POS_LIMIT_135 (2.09439510239 * SAFETY_FACTOR_P)
#define POS_LIMIT_6   (3.05432619099 * SAFETY_FACTOR_P)
// From KukaSim
#define VEL_LIMIT_01  (1.483529 * SAFETY_FACTOR_V) // 85°/s in rad/s
#define VEL_LIMIT_2   (1.745329 * SAFETY_FACTOR_V) // 100°/s in rad/s
#define VEL_LIMIT_3   (1.308996 * SAFETY_FACTOR_V) // 75°/s in rad/s
#define VEL_LIMIT_4   (2.268928 * SAFETY_FACTOR_V) // 130°/s in rad/s
#define VEL_LIMIT_56  (2.356194 * SAFETY_FACTOR_V) // 135°/s in rad/s
#ifndef R_TL
	#define R_TL 100.0
	#define Q_PL 100.0
	#define Q_VL 100.0
#endif

// include vel and pos limits if desired
#if USE_LIMITS_FLAG
	template <typename T>
	__host__ __device__ __forceinline__
	T getPosLimit(int ind){return static_cast<T>(ind == 6 ? POS_LIMIT_6 : (ind % 2 ? POS_LIMIT_135 : POS_LIMIT_024));}

	template <typename T>
	__host__ __device__ __forceinline__
	T getVelLimit(int ind){return static_cast<T>(ind > 4 ? VEL_LIMIT_56 : (ind == 4 ? VEL_LIMIT_4 : (ind == 3 ? VEL_LIMIT_3 : (ind == 2 ? VEL_LIMIT_2 : VEL_LIMIT_01))));}

	template <typename T>
	__host__ __device__ __forceinline__
	T getTorqueLimit(int ind){return static_cast<T>(TORQUE_LIMIT);}

	template <typename T, int dLevel>
	__host__ __device__ __forceinline__
	T pieceWiseQuadratic(T val, T qStart){
		T delta = abs(val) - qStart; 	bool flag = delta > 0;
		// if inside use linear regime
		if(flag){
			if      (dLevel == 0){return static_cast<T>(abs(val));}
			else if (dLevel == 1){return static_cast<T>((val < 0 ? -1 : 1));}
			else if (dLevel == 2){return static_cast<T>(0);}
			else{printf("Derivative of pieceWiseQuadratic is not implemented beyond level 2\n"); return 0;}
		}
		// else quadratic regime
		else{
			if      (dLevel == 0){return static_cast<T>(qStart + static_cast<T>(0.5)*delta*delta);}
			else if (dLevel == 1){return static_cast<T>((val < 0 ? -delta : delta));}
			else if (dLevel == 2){return static_cast<T>(1);}
			else{printf("Derivative of pieceWiseQuadratic is not implemented beyond level 2\n"); return 0;}
		}
	}

	template <typename T, int dLevel>
	__host__ __device__ __forceinline__
	T quadPen(T val, T absMax){
		T delta = abs(val) - absMax;
		if(delta < 0){return 0;}
		else{
			if      (dLevel == 0){return static_cast<T>(0.5*delta*delta);}
			else if (dLevel == 1){return static_cast<T>((val < 0 ? -delta : delta));}
			else if (dLevel == 2){return static_cast<T>(1);}
			else{printf("Derivative of quadPen is not implemented beyond level 2\n"); return 0;}
		}
	}

	template <typename T>
	__host__ __device__ __forceinline__
	void getLimitVars(T *s_x, T *s_u, T *qr, T *val, T *limit, int ind, int k){
		if (ind < NUM_POS){	         *qr = static_cast<T>(Q_PL);		*val = s_x[ind];				*limit = getPosLimit<T>(ind);}
		else if (ind < STATE_SIZE_PDDP){  *qr = static_cast<T>(Q_VL);		*val = s_x[ind];				*limit = getVelLimit<T>(ind-NUM_POS);}
		else{                        *qr = static_cast<T>(R_TL);		*val = s_u[ind-STATE_SIZE_PDDP]; 	*limit = getTorqueLimit<T>(ind-STATE_SIZE_PDDP);}
	}

	template <typename T, int dLevel>
	__host__ __device__ __forceinline__
	T limitCosts(T *s_x, T *s_u, int ind, int k){
		T qr;	T val;	T limit;	getLimitVars(s_x,s_u,&qr,&val,&limit,ind,k);
		// return qr*pieceWiseQuadratic<T,dLevel>(val,limit);
		return qr*quadPen<T,dLevel>(val,limit);
	}
#endif

// default cost multipliers
#ifndef _Q1 // For standrd costs
 	#define _Q1 0.1 // q
	#define _Q2 0.001 // qd
	#define _R  0.0001
	#define _QF1 1000.0 // q
	#define _QF2 1000.0 // qd
	#endif
#ifndef _Q_EE1 // For EE Costs
	#define _Q_EE1 0.1		 // xyz
	#define _Q_EE2 0		 // rpy
	#define _R_EE 0.0001
	#define _QF_EE1 1000.0
	#define _QF_EE2 0
	#define _Q_xdEE 0.1
	#define _QF_xdEE 1000.0
	#define _Q_xEE 0.0
	#define _QF_xEE 0.0
#endif
#ifndef _Q_EEV1 // for EE_Vel Costs
	#define _Q_EEV1 0
	#define _Q_EEV2 0
	#define _QF_EEV1 0
	#define _QF_EEV2 0
#endif
#ifndef SMOOTH_ABS_ALPHA
	#define SMOOTH_ABS_ALPHA 0.2
#endif

// joint level costs are simple
#if !EE_COST_PDDP
	// joint level cost func returns single val
	template <typename T>
	__host__ __device__ __forceinline__
	T costFunc(T *xk, T *uk, T *xgk, int k, T Q1 = _Q1, T Q2 = _Q2, T R = _R, T QF1 = _QF1, T QF2 = _QF2){
		T cost = 0.0;
		if (k == NUM_TIME_STEPS - 1){
			#pragma unroll
	    	for (int i=0; i<STATE_SIZE_PDDP; i++){T delta = xk[i]-xgk[i]; cost += (T) (i < NUM_POS ? QF1 : QF2)*delta*delta;}
    		cost = static_cast<T>(0.5)*cost; // multiply by 1/2 all at once to save cycles
    		#if USE_LIMITS_FLAG
	    		#pragma unroll
    			for (int i=0; i<STATE_SIZE_PDDP; i++){cost += limitCosts<T,0>(xk,uk,i,k);}
    		#endif
	    }
	    else{
	    	#pragma unroll
	        for (int i=0; i<STATE_SIZE_PDDP; i++){T delta = xk[i]-xgk[i]; cost += (T) (i < NUM_POS ? Q1 : Q2)*delta*delta;}
	    	#pragma unroll
	        for (int i=0; i<CONTROL_SIZE; i++){cost += (T) R*uk[i]*uk[i];}
        	cost = static_cast<T>(0.5)*cost; // multiply by 1/2 all at once to save cycles
        	#if USE_LIMITS_FLAG
	        	#pragma unroll
    			for (int i=0; i<STATE_SIZE_PDDP+CONTROL_SIZE; i++){cost += limitCosts<T,0>(xk,uk,i,k);}
    		#endif
		}
		return cost;
	}

	// joint level cost grad
	template <typename T>
	__host__ __device__ __forceinline__
	void costGrad(T *Hk, T *gk, T *xk, T *uk, T *xgk, int k, int ld_H, T Q1 = _Q1, T Q2 = _Q2, T R = _R, T QF1 = _QF1, T QF2 = _QF2){
		if (k == NUM_TIME_STEPS - 1){
			#pragma unroll
	      	for (int i=0; i<STATE_SIZE_PDDP; i++){
	      		#pragma unroll
	         	for (int j=0; j<STATE_SIZE_PDDP; j++){
	            	Hk[i*ld_H + j] = (i != j) ? static_cast<T>(0) : (i < NUM_POS ? QF1 : QF2);
	         	}  
	      	}
	      	#pragma unroll
	      	for (int i=0; i<STATE_SIZE_PDDP; i++){
	         	gk[i] = (i < NUM_POS ? QF1 : QF2)*(xk[i]-xgk[i]);
	      	}
	      	#pragma unroll
	      	for (int i=0; i<CONTROL_SIZE; i++){
	         	gk[i+STATE_SIZE_PDDP] = 0;
	      	}
	      	// add on any limit costs if needed
		  	#if USE_LIMITS_FLAG
		    	#pragma unroll
    			for (int i=0; i<STATE_SIZE_PDDP; i++){gk[i] += limitCosts<T,1>(xk,uk,i,k);}
	  		#endif
	   	}
	   	else{
	      	#pragma unroll
	      	for (int i=0; i<STATE_SIZE_PDDP+CONTROL_SIZE; i++){
	      		#pragma unroll
	         	for (int j=0; j<STATE_SIZE_PDDP+CONTROL_SIZE; j++){
	            	Hk[i*ld_H + j] = (i != j) ? static_cast<T>(0) : (i < NUM_POS ? Q1 : (i < STATE_SIZE_PDDP ? Q2 : R));
	         	}  
	      	}
	      	#pragma unroll
	      	for (int i=0; i<STATE_SIZE_PDDP; i++){
	         	gk[i] = (i < NUM_POS ? Q1 : Q2)*(xk[i]-xgk[i]);
	      	}
	      	#pragma unroll
	      	for (int i=0; i<CONTROL_SIZE; i++){
	         	gk[i+STATE_SIZE_PDDP] = R*uk[i];
	      	}
	      	#if USE_LIMITS_FLAG
		    	#pragma unroll
    			for (int i=0; i<STATE_SIZE_PDDP+CONTROL_SIZE; i++){gk[i] += limitCosts<T,1>(xk,uk,i,k);}
	  		#endif
	   	}
	}

// else need to consider multiple scenarios for end effector costs
#else
	template <typename T>
	__host__ __device__ __forceinline__
	T eeCost(T *s_eePos, T *d_eeGoal, int k, T *s_eeVel = nullptr, T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2,
											 T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, int timeShift = 0){
		T cost = 0;
		unsigned goal_offset = USE_TRACKING_COST ? k*6 : 0;
	 	for (int i = 0; i < 6; i ++){
	    	T delta = s_eePos[i] - d_eeGoal[i+goal_offset]; 
	    	bool flag = k >= NUM_TIME_STEPS-1-timeShift;
	    	cost += static_cast<T>(0.5)*(flag ? (i < 3 ? QF_EE1 : QF_EE2) : (i < 3 ? Q_EE1 : Q_EE2))*delta*delta;
	    	#if USE_EE_VEL_COST
    			if (s_eeVel != nullptr){cost += static_cast<T>(0.5)*(flag ? (i < 3 ? QF_EEV1 : QF_EEV2) : (i < 3 ? Q_EEV1 : Q_EEV2))*s_eeVel[i]*s_eeVel[i];}
    		#endif
	 	}
	 	#if USE_SMOOTH_ABS
	    	cost = (T) sqrt(2*cost + static_cast<T>(SMOOTH_ABS_ALPHA*SMOOTH_ABS_ALPHA)) - static_cast<T>(SMOOTH_ABS_ALPHA);
	 	#endif
	 	return cost;
	}

	template <typename T>
	__host__ __device__ __forceinline__
	T deeCost(T *s_eePos, T *s_deePos, T *d_eeGoal, int k, int r, T *s_eeVel = nullptr, T *s_deePosVel = nullptr, T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, 
											 					  T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, int timeShift = 0){
		T val = 0;	T deePos;	bool flag = k >= NUM_TIME_STEPS-1-timeShift;
		unsigned goal_offset = USE_TRACKING_COST ? k*6 : 0;
	 	#pragma unroll
	 	for (int i = 0; i < 6; i++){
	 		T delta = s_eePos[i]-d_eeGoal[i+goal_offset];
	    	#if USE_EE_VEL_COST
    			if (s_eeVel != nullptr){
    				T deeVel = s_deePosVel[r*12+i+6];	deePos = s_deePosVel[r*12+i];
    				val += (flag ? (i < 3 ? QF_EEV1 : QF_EEV2) : (i < 3 ? Q_EEV1 : Q_EEV2))*s_eeVel[i]*deeVel;
    			}
			#else
    			deePos = s_deePos[r*6+i];
			#endif
    		val += (flag ? (i < 3 ? QF_EE1 : QF_EE2) : (i < 3 ? Q_EE1 : Q_EE2))*delta*deePos;
	 	}
	 	#if USE_SMOOTH_ABS
			T val2 = 0;
	    	#pragma unroll
	    	for (int i = 0; i < 6; i++){
	    		T delta = s_eePos[i]-d_eeGoal[i+goal_offset];
	       		val2 += (flag ? (i < 3 ? QF_EE1 : QF_EE2) : (i < 3 ? Q_EE1 : Q_EE2))*delta*delta;
       			if (USE_EE_VEL_COST && s_eeVel != nullptr){val2 += (flag ? (i < 3 ? QF_EEV1 : QF_EEV2) : (i < 3 ? Q_EEV1 : Q_EEV2))*s_eeVel[i]*s_eeVel[i];}
	    	}
	    	val2 += static_cast<T>(SMOOTH_ABS_ALPHA*SMOOTH_ABS_ALPHA);
	    	val /= sqrt(val2);
	 	#endif
	 	return val;
	}

	template <typename T>
	__host__ __device__ __forceinline__
	T nominalStateCost(T *s_x, int ind, int k, T *xTarget = nullptr, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE){
		T Qq = (k == NUM_TIME_STEPS-1 ? QF_xEE : Q_xEE); T Qqd = (k == NUM_TIME_STEPS-1 ? QF_xdEE : Q_xdEE); T deltaq, deltaqd;
		if (xTarget == nullptr){deltaq = s_x[ind];	deltaqd = s_x[ind+NUM_POS];}
		else{deltaq = s_x[ind] - xTarget[ind];	deltaqd = s_x[ind + NUM_POS] - xTarget[ind+NUM_POS];}
		return static_cast<T>(0.5)*(Qq*deltaq*deltaq + Qqd*deltaqd*deltaqd);
	}

	template <typename T, int dLevel>
	__host__ __device__ __forceinline__
	T dNominalStateCost(T *s_x, int ind, int k, T *xTarget = nullptr, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE){
		T Q = (ind < NUM_POS) ? (k == NUM_TIME_STEPS-1 ? QF_xEE : Q_xEE) : (k == NUM_TIME_STEPS-1 ? QF_xdEE : Q_xdEE);
		if (dLevel == 1){T x = s_x[ind]; T delta = (xTarget == nullptr) ? x : x - xTarget[ind]; return Q*delta;}
		else if (dLevel == 2){return Q;}
		else{printf("Derivative for nominal state cost not defined past dLevel = 2\n"); return 0;}
	}

	// eeCost Func to split shared mem
	template <typename T>
	__host__ __device__ __forceinline__
	void costFunc(T *s_cost, T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k, T *s_eeVel = nullptr, 
				  T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, 
				  T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, 
				  T R_EE = _R_EE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, 
				  int timeShift = 0, T *xTarget = nullptr){
		int start, delta; singleLoopVals(&start,&delta);
		#pragma unroll
	    for (int ind = start; ind < NUM_POS; ind += delta){
	    	T cost = 0;
	    	if(ind == 0){cost += eeCost<T>(s_eePos,d_eeGoal,k,s_eeVel,Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,timeShift);} // compute in one thread incase smooth abs (for EEcost)
	      	cost += static_cast<T>(0.5)*(k == NUM_TIME_STEPS-1 ? static_cast<T>(0) : R_EE)*s_u[ind]*s_u[ind]; // add on input cost
	      	cost += nominalStateCost<T>(s_x,ind,k,xTarget,Q_xEE,QF_xEE,Q_xdEE,QF_xdEE); // add on the nominal state target cost
	      	#if USE_LIMITS_FLAG // add on any limit costs if needed
	      		cost += limitCosts<T,0>(s_x,s_u,ind,k); cost += limitCosts<T,0>(s_x,s_u,ind+NUM_POS,k); cost += limitCosts<T,0>(s_x,s_u,ind+STATE_SIZE_PDDP,k);
	  		#endif
	      	s_cost[ind] += cost;
	   	}
	}

	// eeCost Func returns single val
	template <typename T>
	__host__ __device__ __forceinline__
	T costFunc(T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k, T *s_eeVel = nullptr,
			   T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, 
			   T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, 
			   T R_EE = _R_EE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, 
			   int timeShift = 0, T *xTarget = nullptr){
		T cost = 0;
		#pragma unroll
	    for (int ind = 0; ind < NUM_POS; ind ++){
	      	if(ind == 0){cost += eeCost<T>(s_eePos,d_eeGoal,k,s_eeVel,Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,timeShift);} // compute in one thread incase smooth abs (for EEcost)
	      	cost += static_cast<T>(0.5)*(k == NUM_TIME_STEPS-1 ? static_cast<T>(0) : R_EE)*s_u[ind]*s_u[ind]; // add on input cost
	      	cost += nominalStateCost<T>(s_x,ind,k,xTarget,Q_xEE,QF_xEE,Q_xdEE,QF_xdEE); // add on the nominal state target cost
	      	#if USE_LIMITS_FLAG // add on any limit costs if needed
	      		cost += limitCosts<T,0>(s_x,s_u,ind,k); cost += limitCosts<T,0>(s_x,s_u,ind+NUM_POS,k); cost += limitCosts<T,0>(s_x,s_u,ind+STATE_SIZE_PDDP,k);
	  		#endif
	   	}
	   	return cost;
	}

	// eeCost Grad
	template <typename T>
	__host__ __device__ __forceinline__
	void costGrad(T *Hk, T*gk, T *s_eePos, T *s_deePos, T *d_eeGoal, T *s_x, T *s_u, int k, int ld_H, 
				  T *d_JT = nullptr, int tid = -1, T *s_eeVel = nullptr, T *s_deePosVel = nullptr,
				  T Q_EE1 = _Q_EE1, T Q_EE2 = _Q_EE2, T QF_EE1 = _QF_EE1, T QF_EE2 = _QF_EE2, 
				  T Q_EEV1 = _Q_EEV1, T Q_EEV2 = _Q_EEV2, T QF_EEV1 = _QF_EEV1, T QF_EEV2 = _QF_EEV2, 
				  T R_EE = _R_EE, T Q_xdEE = _Q_xdEE, T QF_xdEE = _QF_xdEE, T Q_xEE = _Q_xEE, T QF_xEE = _QF_xEE, 
				  int timeShift = 0, T *xTarget = nullptr){
		// then to get the gradient and Hessian we need to compute the following for the state block (and also standard control block)
		// J = \sum_i Q_i*pow(hand_delta_i,2) + other stuff
		// dJ/dx = g = \sum_i Q_i*hand_delta_i*dh_i/dx + other stuff
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		int start, delta; singleLoopVals(&start,&delta);
		#pragma unroll
		for (int r = start; r < DIM_g_r; r += delta){
		  	T val = 0;
		  	#if USE_EE_VEL_COST 
		  		if (s_deePosVel != nullptr && r < STATE_SIZE_PDDP){val += deeCost<T>(s_eePos,s_deePos,d_eeGoal,k,r,s_eeVel,s_deePosVel,Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,timeShift);}
	  		#else
		  		if (r < NUM_POS){val += deeCost<T>(s_eePos,s_deePos,d_eeGoal,k,r,nullptr,nullptr,Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,timeShift);}
	  		#endif
			if (r < STATE_SIZE_PDDP){val += dNominalStateCost<T,1>(s_x,r,k,xTarget,Q_xEE,QF_xEE,Q_xdEE,QF_xdEE);} // nominal state target cost
			else{val += (k == NUM_TIME_STEPS - 1 ? static_cast<T>(0) : R_EE)*s_u[r-STATE_SIZE_PDDP];} // control cost
		  	#if USE_LIMITS_FLAG // add on any limit costs if needed
	      		val += limitCosts<T,1>(s_x,s_u,r,k);
	  		#endif
		  	gk[r] = val;
		}
		hd__syncthreads();
		// d2J/dx2 = H \approx dh_i/dx'*dh_i/dx + other stuff
		#pragma unroll
		for (int c = starty; c < DIM_H_c; c += dy){
		  	T *H = &Hk[c*ld_H];
		  	#pragma unroll
		  	for (int r= startx; r<DIM_H_r; r += dx){
		     	T val = 0;
		     	#if USE_EE_VEL_COST
			     	if (s_deePosVel != nullptr && r < STATE_SIZE_PDDP && c < STATE_SIZE_PDDP){
				     	#pragma unroll
			        	for (int j = 0; j < 12; j++){//for (int j = 0; j < 12; j++){
			        		//T factor = (k == NUM_TIME_STEPS - 1 ? (j < 3 ? QF_EE1 : (j < 6 ? QF_EE2 : (j < 9 ? QF_EEV1 : QF_EEV2))) : (j < 3 ? Q_EE1 : (j < 6 ? Q_EE2 : (j < 9 ? Q_EEV1 : Q_EEV2))));
			        		val += s_deePosVel[r*12+j]*s_deePosVel[c*12+j];//*factor;
			        	}
	        		}
        		#else
	        		if (r < NUM_POS && c < NUM_POS){
		        		#pragma unroll
			           	for (int j = 0; j < 6; j++){
			           		// T factor = (k == NUM_TIME_STEPS - 1 ? (j < 3 ? QF_EE1 : QF_EE2) : (j < 3 ? Q_EE1 : Q_EE2));
			           		val += s_deePos[r*6+j]*s_deePos[c*6+j];//*factor;
			           	}
		           	}
	           	#endif
			    if (r == c){
					if (r < STATE_SIZE_PDDP){val += dNominalStateCost<T,2>(s_x,r,k,xTarget,Q_xEE,QF_xEE,Q_xdEE,QF_xdEE);} // nominal state target cost
		        	else {val += (k== NUM_TIME_STEPS - 1) ? static_cast<T>(0) : R_EE;} // control cost
		        	#if USE_LIMITS_FLAG // add on any limit costs if needed
	      				val += limitCosts<T,2>(s_x,s_u,r,k);
	  				#endif
		     	}
		     	H[r] = val;
		  	}
		}
		//if cost asked for compute it
		bool flag = d_JT != nullptr; int ind = (tid != -1 ? tid : k);
		#ifdef __CUDA_ARCH__
			if(threadIdx.x != 0 || threadIdx.y != 0){flag = 0;}
			if (flag){d_JT[ind] = costFunc(s_eePos,d_eeGoal,s_x,s_u,k,s_eeVel,Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,timeShift,xTarget);}
		#else
			if (flag){d_JT[ind] += costFunc(s_eePos,d_eeGoal,s_x,s_u,k,s_eeVel,Q_EE1,Q_EE2,QF_EE1,QF_EE2,Q_EEV1,Q_EEV2,QF_EEV1,QF_EEV2,R_EE,Q_xdEE,QF_xdEE,Q_xEE,QF_xEE,timeShift,xTarget);}
		#endif
	}
#endif
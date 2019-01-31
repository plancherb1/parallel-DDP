/***
nvcc -std=c++11 -o iLQR.exe WAFR_iLQR_examples.cu utils/cudaUtils.cu utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -rdc=true -O3
***/
#define EE_COST 0
#define TOL_COST 0.0

#include "DDPHelpers.cuh"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

#define TEST_ITERS 100
#define ROLLOUT_FLAG 0
#define RANDOM_MEAN 0.0
#if PLANT == 1 // pend
	#define RANDOM_STDEV 0.001
	#define GOAL_T 3.1416
	#define GOAL_O 0.0
#elif PLANT == 2 // cart
	#define RANDOM_STDEV 0.001
	#define GOAL_X 0.0
	#define GOAL_T 3.1416
 	#define GOAL_O 0.0
#elif PLANT == 3 // quad
	#define RANDOM_STDEV 0.001
 	#define GOAL_X 7.0
	#define GOAL_Y 10.0
	#define GOAL_Z 0.5
	#define GOAL_O 0.0
#elif PLANT == 4 // arm
	#define RANDOM_STDEV 0.001
 	#define PI 3.14159
	#define GOAL_1 0
	#define GOAL_2 0
	#define GOAL_3 0
	#define GOAL_4 -0.25*PI
	#define GOAL_5 0
	#define GOAL_6 0.25*PI
	#define GOAL_7 0.5*PI
	#define GOAL_O 0.0
#endif

char errMsg[]  = "Error: Unkown code - usage is [C]PU or [G]PU with [CS] for serial line search\n";
char tot[]  = " TOT";	char init[] = "INIT";	char fp[]   = "  FP";	char fs[]   = "  FS";	char bp[]   = "  BP";	char nis[]  = " NIS";
double tTime[TEST_ITERS];	double fsimTime[TEST_ITERS*MAX_ITER];	double fsweepTime[TEST_ITERS*MAX_ITER];	double bpTime[TEST_ITERS*MAX_ITER];
double nisTime[TEST_ITERS*MAX_ITER];	double initTime[TEST_ITERS];	algType Jout[TEST_ITERS*(MAX_ITER+1)];	int alphaOut[TEST_ITERS*(MAX_ITER+1)];
std::default_random_engine randEng(time(0)); //seed
std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

template <typename T>
__host__ __forceinline__
void loadXU(T *x, T *u, T *xGoal, int ld_x, int ld_u){
	for (int k=0; k<NUM_TIME_STEPS; k++){
		T *xk = x + k*ld_x;
		#if PLANT == 1 // pend
			xk[0] = 0.0;	xk[1] = (T)randDist(randEng);
		#elif PLANT == 2 // cart
		 	xk[0] = 0.0;				xk[1] = 0.0;
			xk[2] = (T)randDist(randEng);	xk[3] = (T)randDist(randEng);
		#elif PLANT == 3 // quad
			for (int k2=0; k2<STATE_SIZE; k2++){if (k2 == 2){xk[k2] = 0.5;}	else if(k2 >= NUM_POS){xk[k2] = (T)randDist(randEng);}	else{xk[k2] = 0.0;}}
		#elif PLANT == 4 // arm
			xk[0] = (T)-0.5*PI;		xk[1] = (T)0.25*PI;		xk[2] = (T)0.167*PI;
			xk[3] = (T)-0.167*PI;	xk[4] = (T)0.125*PI;	xk[5] = (T)0.167*PI;	xk[6] = (T)0.5*PI;
			xk[7] = (T)randDist(randEng);	xk[8] = (T)randDist(randEng);	xk[9] = (T)randDist(randEng);
			xk[10] = (T)randDist(randEng);	xk[11] = (T)randDist(randEng);	xk[12] = (T)randDist(randEng);	xk[13] = (T)randDist(randEng);
		#endif
	}
	for (int k=0; k<NUM_TIME_STEPS; k++){
		T *uk = u + k*ld_u;
		#if PLANT == 1 || PLANT == 2 // pend and cart
			uk[0] = 0.01;
		#elif PLANT == 3 // quad
			for (int k2=0; k2<CONTROL_SIZE; k2++){uk[k2] = 1.22625;}
		#elif PLANT == 4 // arm
			uk[0] = 0.0;		uk[1] = -102.9832;	uk[2] = 11.1968;
			uk[3] = 47.0724;	uk[4] = 2.5993;		uk[5] = -7.0290;	uk[6] = -0.0907;
		#endif
	}
	#if EE_COST
		#if PLANT == 4 // arm
			const T temp[] = {GOAL_X,GOAL_Y,GOAL_Z,GOAL_r,GOAL_p,GOAL_y};
			for (int i=0; i < 6; i++){xGoal[i] = temp[i];}
		#else
			#error "PLANT DOES NOT HAVE END EFFECTOR -- RUN WITH EE_COST = 0"
		#endif
		
	#else
		#if PLANT == 1 // pend
			const T temp[] = {GOAL_T,GOAL_O};
		#elif PLANT == 2 // cart
		 	const T temp[] = {GOAL_X,GOAL_T,GOAL_O,GOAL_O};
		#elif PLANT == 3 // quad
		 	const T temp[] = {GOAL_X,GOAL_Y,GOAL_Z,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O};
		#elif PLANT == 4 // arm
			const T temp[] = {GOAL_1,GOAL_2,GOAL_3,GOAL_4,GOAL_5,GOAL_6,GOAL_7,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O,GOAL_O};
		#endif
		for (int i=0; i < STATE_SIZE; i++){xGoal[i] = temp[i];}
	#endif
}
__host__
void computeStats(double *_median, double *_avg, double *_stdev, double *_min, double *_max, int size, std::vector<double> v){
	// sort gives us the median, max and min
    std::sort(v.begin(),v.end());
	*_median = size % 2 ? v[size / 2] : (v[size / 2 - 1] + v[size / 2]) / 2.0;	
	*_max = v.back();	
	*_min = v.front();
	// sum gives use the average
	double sum = std::accumulate(v.begin(), v.end(), 0.0);		
	*_avg = sum / (double)size;	
	// and then the std dev	
	*_stdev = 0.0;
	for(std::vector<double>::iterator it = v.begin(); it != v.end(); ++it){
		*_stdev += pow(*it-*_avg,2.0);
	}
	*_stdev /= (double) size;	
	*_stdev = pow(*_stdev,0.5);
}
__host__ 
void printTimingStats(double *arr, int iters, char *type, int vals_per_test = MAX_ITER){
	double _median, _avg, _stdev, _min, _max;	std::vector<double> v;
	// get all the vals into a vector
    for (int test=0; test<TEST_ITERS; test++){for (int i=0; i<iters; i++){v.push_back(arr[test*vals_per_test + i]);}}
    computeStats(&_median, &_avg, &_stdev, &_min, &_max, TEST_ITERS*iters, v);
	// report results
	printf("%s: Median[%f] Average[%f] StdDev[%f] max[%f] min[%f]\n",type,_median,_avg,_stdev,_max,_min);
}
__host__
void printPerIterTiming(double *initTime, double *fsimTime, double *fsweepTime, double *bpTime, double *nisTime){
	// the total time to each cost point is:
	//   t[i+1] = fsim[i] + fsweep[i] + bp[i] + nis[i-1] + t[i]
	//   t[1] = init + fsim[0] + fsweep[0] + bp[0]
	//   t[0] = 0
	double _median, _avg, _stdev, _min, _max;
	double TIMES[MAX_ITER+1];	TIMES[0] = 0.0;
	for (int iter = 0; iter < MAX_ITER; iter++){
		// collect the vector for this time step
		std::vector<double> v;
		for (int test=0; test<TEST_ITERS; test++){
			int ind = test*MAX_ITER + iter;
			double val = fsimTime[ind] + fsweepTime[ind] + bpTime[ind] + (iter > 0 ? nisTime[ind-1]	: initTime[test]);
			v.push_back(val);
		}
		// compute the stats
		computeStats(&_median, &_avg, &_stdev, &_min, &_max, TEST_ITERS, v);
		// printf("Full Timing Per Iter[%d]: Median[%f] Average[%f] StdDev[%f] max[%f] min[%f]\n",iter+1,_median,_avg,_stdev,_max,_min);
		TIMES[iter+1] = _median + TIMES[iter];
	}
	printf("Median Time Trace:\n");
	for (int iter = 0; iter <= MAX_ITER; iter++){printf("%f ",TIMES[iter]);} printf("\n");
}
template <typename T>
__host__ __forceinline__
void printJAlphaStats(T *Jout, int *alphaOut){
	double _median, _avg, _stdev, _min, _max, _median2, _avg2, _stdev2, _min2, _max2;
	std::vector<double> MEDIANS;
	for (int iter = 0; iter <= MAX_ITER; iter++){
		std::vector<double> v;	std::vector<double> v2;
		for (int i=0; i<TEST_ITERS; i++){v.push_back((double)Jout[i*(MAX_ITER+1) + iter]);	v2.push_back((double)alphaOut[i*(MAX_ITER+1) + iter]);}
		computeStats(&_median, &_avg, &_stdev, &_min, &_max, TEST_ITERS, v);
		computeStats(&_median2, &_avg2, &_stdev2, &_min2, &_max2, TEST_ITERS, v2);
		MEDIANS.push_back(_median);
		printf("Iter %d: Median[%f] Average[%f] StdDev[%f] max[%f] min[%f] alpha[%.1f,%.1f,%.1f,%.0f,%.0f]\n",iter,_median,_avg,_stdev,_max,_min,_median2,_avg2,_stdev2,_max2,_min2);
	}
	printf("Median J Trace:\n");
	for(std::vector<double>::iterator it = MEDIANS.begin(); it != MEDIANS.end(); ++it){printf("%f ",*it);}printf("\n");
}

__host__ __forceinline__
bool tryParse(std::string& input, int& output) {
	try{output = std::stoi(input);}
	catch (std::invalid_argument) {return false;}
	return true;
}
__host__ __forceinline__
int getNumIters(int maxInt, int minInt){
   std::string input;	std::string exitCode ("q"); int x;
   printf("Which iter would you like to stop timing on (q to exit)?\n");
   while(1){
      getline(std::cin, input);
      while (!tryParse(input, x)){
         if (input.compare(input.size()-1,1,exitCode) == 0){return -1;}
         std::cout << "Bad entry. Enter a NUMBER\n";	getline(std::cin, input);
      }
      if (x >= minInt && x <= maxInt){break;}
      else{std::cout << "Entry must be in range[" << minInt << "," << maxInt << "]\n";}
   }
   return x;
}

__host__ 
void printAllTimingStats(double *tTime, double *initTime, double *fsimTime, double *fsweepTime, double *bpTime, double *nisTime){
	printPerIterTiming(initTime,fsimTime,fsweepTime,bpTime,nisTime);
	while(1){   
		// ask for which iter to stop timing stats on
		int timingIters = getNumIters(MAX_ITER,1);
		if (timingIters == -1){break;}
		// print those stats
		printTimingStats(tTime,TEST_ITERS,tot,1);
		printTimingStats(initTime,TEST_ITERS,init,1);
		printTimingStats(fsimTime,timingIters,fp);
		printTimingStats(fsweepTime,timingIters,fs);
		printTimingStats(bpTime,timingIters,bp);
		printTimingStats(nisTime,timingIters,nis);
   }
}

template <typename T>
__host__
void testCPU(int serialAlphas){
	// CPU VARS	
	// first integer constants for the leading dimmensions of allocaitons
	int ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A;
	// algorithm hyper parameters
	T *alpha;
	// then variables defined by blocks for backward pass
	T *P, *p, *Pp, *pp, *AB, *H, *g, *KT, *du;
	// variables for forward pass
	T *x, *u, *xp, *xp2, *up, *JT;
	// variables for forward sweep
	T *d, *dp, *ApBK, *Bdu;
	// for checking inversion errors
	int *err;
	// for expected cost reduction
	T *dJexp;
	// goal point
  	T *xGoal;
    // Inertias and Tbodybase
    T *I, *Tbody;
    // parallel alphas
    T **xs, **us, **ds, **JTs;

	// Allocate space and initialize the variables
	if(serialAlphas){
		allocateMemory_CPU<T>(&x, &xp, &xp2, &u, &up, &xGoal, &P, &Pp, &p, &pp, &AB, &H, &g, &KT, &du, &d, &dp, &ApBK, &Bdu, 
   					      &JT, &dJexp, &alpha, &err, &ld_x, &ld_u, &ld_P, &ld_p, &ld_AB, &ld_H, &ld_g, &ld_KT, &ld_du, &ld_d, &ld_A,
                          &I, &Tbody);	
	}
    else{
    	allocateMemory_CPU2<T>(&xs, &xp, &xp2, &us, &up, &xGoal, &P, &Pp, &p, &pp, &AB, &H, &g, &KT, &du, &ds, &dp, &ApBK, &Bdu, 
   						       &JTs, &dJexp, &alpha, &err, &ld_x, &ld_u, &ld_P, &ld_p, &ld_AB, &ld_H, &ld_g, &ld_KT, &ld_du, &ld_d, &ld_A,
                    	       &I, &Tbody);	
    }

    T *x0 = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
    T *u0 = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));

    for (int i=0; i < TEST_ITERS; i++){
		printf("<<<TESTING CPU %d/%d>>>\n",i+1,TEST_ITERS);
      	loadXU<T>(x0,u0,xGoal,ld_x,ld_u);
      	if(serialAlphas){
      		runiLQR_CPU<T>(x0, u0, nullptr, nullptr, nullptr, nullptr, xGoal, &Jout[i*(MAX_ITER+1)], &alphaOut[i*(MAX_ITER+1)], ROLLOUT_FLAG, 1, 1,
					       &tTime[i], &fsimTime[i*MAX_ITER], &fsweepTime[i*MAX_ITER], &bpTime[i*MAX_ITER], &nisTime[i*MAX_ITER], &initTime[i], 
					       x, xp, xp2, u, up, P, p, Pp, pp, AB, H, g, KT, du, d, dp, ApBK, Bdu, alpha, JT, dJexp, err,
					       ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A, I, Tbody);
      	}
      	else{
	      	runiLQR_CPU2<T>(x0, u0, nullptr, nullptr, nullptr, nullptr, xGoal, &Jout[i*(MAX_ITER+1)], &alphaOut[i*(MAX_ITER+1)], ROLLOUT_FLAG, 1, 1,
						    &tTime[i], &fsimTime[i*MAX_ITER], &fsweepTime[i*MAX_ITER], &bpTime[i*MAX_ITER], &nisTime[i*MAX_ITER], &initTime[i], 
						    xs, xp, xp2, us, up, P, p, Pp, pp, AB, H, g, KT, du, ds, dp, ApBK, Bdu, alpha, JTs, dJexp, err,
						    ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A, I, Tbody);	
      	}
	}
	// print final state
	printf("Final state:\n");	for (int i = 0; i < STATE_SIZE; i++){printf("%15.5f ",x0[(NUM_TIME_STEPS-2)*ld_x + i]);}	printf("\n");
	//printf("Final utraj:\n");    for (int i = 0; i < NUM_TIME_STEPS-1; i++){printMat<T,1,DIM_u_r>(&u0[i*ld_u],1);}
	
	// print all requested statistics
   	printJAlphaStats(Jout,alphaOut);
   	printAllTimingStats(tTime,initTime,fsimTime,fsweepTime,bpTime,nisTime);
	
	// free those vars
	if(serialAlphas){freeMemory_CPU<T>(x, xp, xp2, u, up, P, Pp, p, pp, AB, H, g, KT, du, d, dp, Bdu, ApBK, dJexp, err, alpha, JT, xGoal, I, Tbody);}
	else{freeMemory_CPU2<T>(xs, xp, xp2, us, up, P, Pp, p, pp, AB, H, g, KT, du, ds, dp, Bdu, ApBK, dJexp, err, alpha, JTs, xGoal, I, Tbody);}
	free(x0);
    free(u0);
}

template <typename T>
__host__
void testGPU(){
	// GPU VARS	
	// first integer constants for the leading dimmensions of allocaitons
	int ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A;
	// then vars for stream handles
	cudaStream_t *streams;
	// algorithm hyper parameters
	T *alpha, *d_alpha;
	int *alphaIndex;
	// then variables defined by blocks for backward pass
	T *d_P, *d_p, *d_Pp, *d_pp, *d_AB, *d_H, *d_g, *d_KT, *d_du;
	// variables for forward pass
	T **d_x, **d_u, **h_d_x, **h_d_u, *d_xp, *d_xp2, *d_up, *d_JT, *J;
	// variables for forward sweep
	T **d_d, **h_d_d, *d_dp, *d_dT, *d, *d_ApBK, *d_Bdu, *d_dM;
	// for checking inversion errors
	int *err, *d_err;
	// for expected cost reduction
	T *dJexp, *d_dJexp;
	// goal point
	T *xGoal, *d_xGoal;
    // Inertias and Tbodybase
    T *d_I, *d_Tbody;

	// Allocate space and initialize the variables
   	allocateMemory_GPU<T>(&d_x, &h_d_x, &d_xp, &d_xp2, &d_u, &h_d_u, &d_up, &d_xGoal, &xGoal,
   					   &d_P, &d_Pp, &d_p, &d_pp, &d_AB, &d_H, &d_g, &d_KT, &d_du,
   					   &d_d, &h_d_d, &d_dp, &d_dT, &d_dM, &d, &d_ApBK, &d_Bdu,
   					   &d_JT, &J, &d_dJexp, &dJexp, &alpha, &d_alpha, &alphaIndex, &d_err, &err, 
   					   &ld_x, &ld_u, &ld_P, &ld_p, &ld_AB, &ld_H, &ld_g, &ld_KT, &ld_du, &ld_d, &ld_A,
                       &streams, &d_I, &d_Tbody);

   	T *x0 = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
   	T *u0 = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));

   	for (int i=0; i<TEST_ITERS; i++){
		printf("<<<TESTING GPU %d/%d>>>\n",i+1,TEST_ITERS);
		loadXU<T>(x0,u0,xGoal,ld_x,ld_u);
      	runiLQR_GPU<T>(x0, u0, nullptr, nullptr, nullptr, nullptr, xGoal, &Jout[i*(MAX_ITER+1)], &alphaOut[i*(MAX_ITER+1)], ROLLOUT_FLAG, 1,  1,
      				&tTime[i], &fsimTime[i*MAX_ITER], &fsweepTime[i*MAX_ITER], &bpTime[i*MAX_ITER], &nisTime[i*MAX_ITER], &initTime[i], streams,
					d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, d_P, d_p, d_Pp, d_pp, d_AB, d_H, d_g, d_KT, d_du,
					d_d, h_d_d, d_dp, d_dT, d, d_ApBK, d_Bdu, d_dM, alpha, d_alpha, alphaIndex, d_JT, J, dJexp, d_dJexp, d_xGoal,
					err, d_err, ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A, d_I, d_Tbody);
   	}
   	// print final state
	printf("Final state:\n");	for (int i = 0; i < STATE_SIZE; i++){printf("%15.5f ",x0[(NUM_TIME_STEPS-2)*ld_x + i]);}	printf("\n");
	// printf("Final xtraj:\n");   for (int i = 0; i < NUM_TIME_STEPS; i++){printMat<T,1,DIM_x_r>(&x0[i*ld_x],1);}

	// print all requested statistics
   	printJAlphaStats(Jout,alphaOut);
   	printAllTimingStats(tTime,initTime,fsimTime,fsweepTime,bpTime,nisTime);

	// free those vars
	freeMemory_GPU<T>(d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, xGoal, d_xGoal,  d_P, d_Pp, d_p, d_pp, d_AB, d_H, d_g, d_KT, d_du, 
				   d_d, h_d_d, d_dp, d_dM, d_dT, d,  d_ApBK, d_Bdu, d_JT, J, d_dJexp, dJexp, alpha, d_alpha, alphaIndex, d_err, err, 
                   streams, d_I, d_Tbody);
	free(x0);
   	free(u0);
}

template <typename T>
__host__
void testGPU_SLQ(){
	// GPU VARS	
	// first integer constants for the leading dimmensions of allocaitons
	int ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A;
	// then vars for stream handles
	cudaStream_t *streams;
	// algorithm hyper parameters
	T *alpha, *d_alpha;
	int *alphaIndex;
	// then variables defined by blocks for backward pass
	T *d_P, *d_p, *d_Pp, *d_pp, *d_AB, *d_H, *d_g, *d_KT, *d_du;
	// variables for forward pass
	T **d_x, **d_u, **h_d_x, **h_d_u, *d_xp, *d_xp2, *d_up, *d_JT, *J;
	// variables for forward sweep
	T **d_d, **h_d_d, *d_dp, *d_dT, *d, *d_ApBK, *d_Bdu, *d_dM;
	// for checking inversion errors
	int *err, *d_err;
	// for expected cost reduction
	T *dJexp, *d_dJexp;
	// goal point
	T *xGoal, *d_xGoal;
    // Inertias and Tbodybase
    T *d_I, *d_Tbody;

	// Allocate space and initialize the variables
   	allocateMemory_GPU<T>(&d_x, &h_d_x, &d_xp, &d_xp2, &d_u, &h_d_u, &d_up, &d_xGoal, &xGoal,
   					   &d_P, &d_Pp, &d_p, &d_pp, &d_AB, &d_H, &d_g, &d_KT, &d_du,
   					   &d_d, &h_d_d, &d_dp, &d_dT, &d_dM, &d, &d_ApBK, &d_Bdu,
   					   &d_JT, &J, &d_dJexp, &dJexp, &alpha, &d_alpha, &alphaIndex, &d_err, &err, 
   					   &ld_x, &ld_u, &ld_P, &ld_p, &ld_AB, &ld_H, &ld_g, &ld_KT, &ld_du, &ld_d, &ld_A,
                       &streams, &d_I, &d_Tbody);

   	T *x0 = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
   	T *u0 = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));

   	for (int i=0; i<TEST_ITERS; i++){
		printf("<<<TESTING GPU SLQ %d/%d>>>\n",i+1,TEST_ITERS);
		loadXU<T>(x0,u0,xGoal,ld_x,ld_u);
      	runSLQ_GPU<T>(x0, u0, nullptr, nullptr, nullptr, nullptr, xGoal, &Jout[i*(MAX_ITER+1)], &alphaOut[i*(MAX_ITER+1)], ROLLOUT_FLAG, 1,  1,
      				&tTime[i], &fsimTime[i*MAX_ITER], &fsweepTime[i*MAX_ITER], &bpTime[i*MAX_ITER], &nisTime[i*MAX_ITER], &initTime[i], streams,
					d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, d_P, d_p, d_Pp, d_pp, d_AB, d_H, d_g, d_KT, d_du,
					d_d, h_d_d, d_dp, d_dT, d, d_ApBK, d_Bdu, d_dM, alpha, d_alpha, alphaIndex, d_JT, J, dJexp, d_dJexp, d_xGoal,
					err, d_err, ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A, d_I, d_Tbody);
   	}
   	// print final state
	printf("Final state:\n");	for (int i = 0; i < STATE_SIZE; i++){printf("%15.5f ",x0[(NUM_TIME_STEPS-2)*ld_x + i]);}	printf("\n");
	// printf("Final xtraj:\n");   for (int i = 0; i < NUM_TIME_STEPS; i++){printMat<T,1,DIM_x_r>(&x0[i*ld_x],1);}

	// print all requested statistics
   	printJAlphaStats(Jout,alphaOut);
   	printAllTimingStats(tTime,initTime,fsimTime,fsweepTime,bpTime,nisTime);

	// free those vars
	freeMemory_GPU<T>(d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, xGoal, d_xGoal,  d_P, d_Pp, d_p, d_pp, d_AB, d_H, d_g, d_KT, d_du, 
				   d_d, h_d_d, d_dp, d_dM, d_dT, d,  d_ApBK, d_Bdu, d_JT, J, d_dJexp, dJexp, alpha, d_alpha, alphaIndex, d_err, err, 
                   streams, d_I, d_Tbody);
	free(x0);
   	free(u0);
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	// test based on command line args
	char hardware = '?'; // require user input
	if (argc > 1){
		hardware = argv[1][0];
	}
	if (hardware == 'G'){testGPU<algType>();}
	else if (hardware == 'C'){int flag = (int)(argv[1][1] == 'S'); testCPU<algType>(flag);}
	else if (hardware == 'S'){testGPU_SLQ<algType>();}
	else{printf("%s",errMsg);}
	return 0;
}

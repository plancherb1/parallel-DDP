#ifndef _EXAMPLE_UTILS_
#define _EXAMPLE_UTILS_
/*****************************************************************
 * Utils for Examples
 * 1 IO, initialization, and error computation helpers
 *****************************************************************/

#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#define PI 3.14159

#if MPC_MODE
	__host__
	bool tryParse(std::string& input, int& output) {
		try{output = std::stoi(input);}
		catch (std::invalid_argument) {return false;}
		return true;
	}
	__host__
	int getInt(int maxInt, int minInt){
		std::string input;	std::string exitCode ("q"); int x;
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
	void keyboardHold(){
	   	printf("Press enter to continue\n");	std::string input;	getline(std::cin, input);
	}
	template <typename T>
	__host__
	void loadInitialState(T *xInit, int mode = 0){
		// mode 0 = vertial position, mode 1 = center of workspace, mode 2 = patrick position
		for (int i = 0; i < STATE_SIZE; i++){xInit[i] = 0;}
		if (mode == 1){xInit[1] = PI/4.0; xInit[3] = -PI/4.0; xInit[5] = PI/4.0;}
		if (mode == 2){xInit[0] = PI/2.0; xInit[1] = -PI/6.0; xInit[2] = -PI/3.0; xInit[3] = -PI/2.0; xInit[4] = 3.0*PI/4.0; xInit[5] = -PI/4.0; xInit[6] = 0.0;}
	}
	template <typename T>
	__host__
	void loadTraj(trajVars<T> *tvars, matDimms *dimms, T *xInit = nullptr, T *uInit = nullptr){
		T *xk = tvars->x;	T *uk = tvars->u;
		for (int k=0; k<NUM_TIME_STEPS; k++){
			for (int i = 0; i < STATE_SIZE; i++){
				xk[i] = (xInit != nullptr) ? xInit[i] : 0.0;	
				if (i < CONTROL_SIZE){uk[i] = (uInit != nullptr) ? uInit[i] : 0.01;}
			}
			xk += (dimms->ld_x);	uk += (dimms->ld_u);
		}
		memset(tvars->KT, 0, (dimms->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
	}
	template <typename T>
	__host__
	void loadTraj(CPUVars<T> *cvars, trajVars<T> *tvars, matDimms *dimms, T *xInit = nullptr, T *uInit = nullptr, bool parallelLineSearch = 0){
			loadTraj<T>(tvars, dimms, xInit, uInit);		memcpy(cvars->xActual, tvars->x, STATE_SIZE*sizeof(T));
			memcpy(cvars->x, tvars->x, (dimms->ld_x)*NUM_TIME_STEPS*sizeof(T));
			memcpy(cvars->u, tvars->u, (dimms->ld_u)*NUM_TIME_STEPS*sizeof(T));
			if (parallelLineSearch){
				for (int i = 0; i < NUM_ALPHA; i++){
					memcpy(cvars->xs[i], tvars->x, (dimms->ld_x)*NUM_TIME_STEPS*sizeof(T));
					memcpy(cvars->us[i], tvars->u, (dimms->ld_u)*NUM_TIME_STEPS*sizeof(T));
				}
			}
	}
	template <typename T>
	__host__
	void loadTraj(GPUVars<T> *gvars, trajVars<T> *tvars, matDimms *dimms, T *xInit = nullptr, T *uInit = nullptr){
		loadTraj<T>(tvars, dimms, xInit, uInit);		memcpy(gvars->xActual, tvars->x, STATE_SIZE*sizeof(T));
		for (int i = 0; i < NUM_ALPHA; i++){
			gpuErrchk(cudaMemcpy(gvars->h_d_x[i], tvars->x, (dimms->ld_x)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gvars->h_d_u[i], tvars->u, (dimms->ld_u)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice));
		}
	}
#endif

#if EE_COST
	template <typename T>
	__host__
	void evNorm(T *xActual, T *xGoal, T *eNorm, T *vNorm, T *eePos){
		compute_eePos_scratch<T>(xActual, &eePos[0]);
		*eNorm = static_cast<T>(sqrt(pow(eePos[0]-xGoal[0],2) + pow(eePos[1]-xGoal[1],2) + pow(eePos[2]-xGoal[2],2)));
		*vNorm = 0; for(int i=0;i<NUM_POS;i++){*vNorm+=(T)pow(xActual[NUM_POS+i],2);} *vNorm = static_cast<T>(sqrt(*vNorm));
	}
	template <typename T>
	__host__
	void evNorm(T *xActual, T *xGoal, T *eNorm, T *vNorm){T eePos[NUM_POS];   evNorm(xActual,xGoal,eNorm,vNorm,eePos);}
#endif

#if defined(USE_LCM) && USE_LCM == 1
	template <typename T>
	__host__
	void evNorm(const drake::lcmt_iiwa_status *msg, T *xGoal, T *eNorm, T *vNorm){
		T xActual[STATE_SIZE];
		// first cast the pos and vel to the right type
		for (int i = 0; i < NUM_POS; i++){xActual[i]         = static_cast<T>(msg->joint_position_measured[i]);}
		for (int i = 0; i < NUM_POS; i++){xActual[i+NUM_POS] = static_cast<T>(msg->joint_velocity_estimated[i]);}
		// then compute the error norms
		evNorm(xActual,xGoal,eNorm,vNorm);
	}
	template <typename T>
	__host__
	void runPrinter(char type){
		lcm::LCM lcm_ptr;	if(!lcm_ptr.good()){printf("LCM Failed to Init\n");} 
		if(type == 'S'){
			LCM_IIWA_STATUS_printer<T> *shandler = new LCM_IIWA_STATUS_printer<T>;
			run_IIWA_STATUS_printer<T>(&lcm_ptr,shandler);
			delete shandler;	
		}
		else if(type == 'C'){
			LCM_IIWA_COMMAND_printer<T> *chandler = new LCM_IIWA_COMMAND_printer<T>;
			run_IIWA_COMMAND_printer<T>(&lcm_ptr,chandler);
			delete chandler;
		}
		else if(type == 'T'){
			LCM_traj_printer<T> *thandler = new LCM_traj_printer<T>;
			run_traj_printer<T>(&lcm_ptr,thandler);
			delete thandler;
		}
		else if(type == 'F'){
			LCM_IIWA_STATUS_FILTERED_printer<T> *fhandler = new LCM_IIWA_STATUS_FILTERED_printer<T>;
			run_IIWA_STATUS_FILTERED_printer<T>(&lcm_ptr,fhandler);
			delete fhandler;
		}
		else{printf("Invalid printer requested as second char [%c]. Currently supports: [S]tatus, [C]ommand, [T]rajectory, or [F]iltered Status\n",type);}
	}
	template <typename T>
	__host__
	int runOtherOptions(char mode, T *xInit, char **argv){
		// run the simulator
		if (mode == 'S'){runLCMSimulator<T>(xInit);}
		// run various printers
		else if (mode == 'P'){runPrinter<T>(argv[1][1]);}
		// else error
		else{printf("Error: Unkown code - usage is: [C]PU or [G]PU MPC Algorithm, Debug [P]rinters, or Kuka [S]imulator\n"); mode = '?';}
		return (mode == '?');
	}
#endif

#endif
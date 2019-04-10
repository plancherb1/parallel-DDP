/***
nvcc -std=c++11 -o pickNPlace.exe LCM_pickNPlace_examples.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -llcm -gencode arch=compute_61,code=sm_61 -rdc=true -O3
***/
#define EE_COST 0
#define USE_WAFR_URDF 1
#define Q1 0.1 // q
#define Q2 0.001 // qd
#define R  0.0001
#define QF1 1000.0 // q
#define QF2 1000.0 // qd

#define MPC_MODE 1
#define USE_LCM 1
#define USE_VELOCITY_FILTER 0
#define IGNORE_MAX_ROX_EXIT 0
#define TOL_COST 0.00001
#define PLANT 4

#define PI 3.14159
#include "../config.cuh"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

__host__ __forceinline__
bool tryParse(std::string& input, int& output) {
	try{output = std::stoi(input);}
	catch (std::invalid_argument) {return false;}
	return true;
}
__host__ __forceinline__
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

__host__ __forceinline__
int getTimeBudget(int maxInt, int minInt){
   printf("What should the MPC time budget be (in ms)? (q to exit)?\n");
   return getInt(maxInt,minInt);
}
__host__ __forceinline__
int getMaxIters(int maxInt, int minInt){
   printf("What is the maximum number of iterations a solver can take? (q to exit)?\n");
   return getInt(maxInt,minInt);
}
__host__ __forceinline__
void keyboardHold(){
   	printf("Press enter to continue\n");	std::string input;	getline(std::cin, input);
}
template <typename T>
__host__ __forceinline__
void loadX(T *xk){
	xk[0] = PI/2.0; 	xk[1] = -PI/6.0; 	xk[2] = -PI/3.0; 	xk[3] = -PI/2.0; 	xk[4] = 3.0*PI/4.0; 	xk[5] = -PI/4.0; 	xk[6] = 0.0;
	for(int i = NUM_POS; i < STATE_SIZE; i++){xk[i] = 0.0;}
}
template <typename T>
__host__ __forceinline__
void loadTraj(T *x, T *u, T *KT, int ld_x, int ld_u, int ld_KT){
	T *xk = &x[0];	T *uk = &u[0];
	for (int k=0; k<NUM_TIME_STEPS; k++){
		loadX<T>(xk);	for(int i = 0; i < CONTROL_SIZE; i++){uk[i] = 0.0;}	xk += ld_x;	uk += ld_u;
	}
	memset(KT, 0, ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
}

template <typename T>
class LCM_PickAndPlaceGoal_Handler {
    public:
    	#define NUM_GOALS 1
    	int goalNum;	double eNormLim;	T goals[NUM_POS*NUM_GOALS];
    	lcm::LCM lcm_ptr; // ptr to LCM object for publish ability

    	/*
    	a = [[-60,55,-25,-65,5,60,0],[65,45,-5,-80,15,30,0],[-30,50,-30,-80,0,55,0],[40,60,35,-80,-135,20,0],[-55,55,20,-60,20,40,0],[80,55,-15,-65,0,30,0]]
		output = ''
		for i in range(len(a)):
			row = a[i]
			for k in range(len(row)):
				val = row[k]/180.0*3.14159
				output += 'goals[' + str(i) + '*NUM_POS + ' + str(k) + '] = ' + str(val) + ';\t'
			output += '\n'
		print output
    	*/

    	LCM_PickAndPlaceGoal_Handler(double eLim) : eNormLim(eLim) {
    		goalNum = 0;		if(!lcm_ptr.good()){printf("LCM Failed to Init in Goal Handler\n");}
    		goals[0] = PI/4.0; goals[1] = PI/3.0; goals[2] = PI/6.0; goals[3] = -PI/3.0; goals[4] = 0.0; goals[5] = PI/4; goals[6] = 0.0;
			// goals[0*NUM_POS + 0] = 1.13446305556;	goals[0*NUM_POS + 1] = 0.7853975;		goals[0*NUM_POS + 2] = -0.08726638889;	goals[0*NUM_POS + 3] = -1.39626222222;	goals[0*NUM_POS + 4] = 0.261799166667;	goals[0*NUM_POS + 5] = 0.523598333333;	goals[0*NUM_POS + 6] = 0.0;
			// goals[1*NUM_POS + 0] = -0.523598333333;	goals[1*NUM_POS + 1] = 0.872663888889;	goals[1*NUM_POS + 2] = -0.523598333333;	goals[1*NUM_POS + 3] = -1.39626222222;	goals[1*NUM_POS + 4] = 0.0;				goals[1*NUM_POS + 5] = 0.959930277778;	goals[1*NUM_POS + 6] = 0.0;	
			// goals[2*NUM_POS + 0] = 0.698131111111;	goals[2*NUM_POS + 1] = 1.04719666667;	goals[2*NUM_POS + 2] = 0.610864722222;	goals[2*NUM_POS + 3] = -1.39626222222;	goals[2*NUM_POS + 4] = -2.3561925;		goals[2*NUM_POS + 5] = 0.349065555556;	goals[2*NUM_POS + 6] = 0.0;
			// goals[3*NUM_POS + 0] = -0.959930277778;	goals[3*NUM_POS + 1] = 0.959930277778;	goals[3*NUM_POS + 2] = 0.349065555556;	goals[3*NUM_POS + 3] = -1.04719666667;	goals[3*NUM_POS + 4] = 0.349065555556;	goals[3*NUM_POS + 5] = 0.698131111111;	goals[3*NUM_POS + 6] = 0.0;	
			// goals[4*NUM_POS + 0] = 1.39626222222;	goals[4*NUM_POS + 1] = 0.959930277778;	goals[4*NUM_POS + 2] = -0.261799166667;	goals[4*NUM_POS + 3] = -1.13446305556;	goals[4*NUM_POS + 4] = 0.0;				goals[4*NUM_POS + 5] = 0.523598333333;	goals[4*NUM_POS + 6] = 0.0;
			// goals[5*NUM_POS + 0] = -1.04719666667;	goals[5*NUM_POS + 1] = 0.959930277778;	goals[5*NUM_POS + 2] = -0.436331944444;	goals[5*NUM_POS + 3] = -1.13446305556;	goals[5*NUM_POS + 4] = 0.0872663888889;	goals[5*NUM_POS + 5] = 1.04719666667;	goals[5*NUM_POS + 6] = 0.0;	
			
    	}
    	~LCM_PickAndPlaceGoal_Handler(){}

    	// load initial goal
    	void loadInitialGoal(T *goal){for(int i = 0; i < NUM_POS; i++){goal[i] = goals[i];}}

		// update goal based on status
		void handleStatus(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
			// compute the position error
			T eNorm;	T *gk = &goals[goalNum*NUM_POS];		const double *qk = &(msg->joint_position_measured[0]);
			for (int i = 0; i < NUM_POS; i++){eNorm += pow(static_cast<T>(qk[i]) - gk[i],2);}
			// debug print
			printf("[%ld] eNorm[%f] for goalNum[%d][%.3f %.3f %.3f %.3f %.3f %.3f %.3f] vs xk[%.3f %.3f %.3f %.3f %.3f %.3f %.3f]\n",
						msg->utime,eNorm,goalNum,gk[0],gk[1],gk[2],gk[3],gk[4],gk[5],gk[6],qk[0],qk[1],qk[2],qk[3],qk[4],qk[5],qk[6]);
			// then figure out if we are ready to change the goal
			if(eNorm < eNormLim){
				// then change goal and publish new goal
				goalNum = (goalNum + 1)%NUM_GOALS;
				kuka::lcmt_target_position dataOut;   dataOut.utime = msg->utime;
				for (int i = 0; i < NUM_POS; i++){dataOut.position[i] = goals[goalNum*NUM_POS + i]; dataOut.velocity[i] = 0;}
				// and publish it to goal channel
			    lcm_ptr.publish(ARM_GOAL_CHANNEL,&dataOut);
			}
		}
};
template <typename T>
void runPickAndPlaceGoalLCM(LCM_PickAndPlaceGoal_Handler<T> *handler){
	lcm::LCM lcm_ptr;
	lcm::Subscription *sub = lcm_ptr.subscribe(ARM_STATUS_FILTERED, &LCM_PickAndPlaceGoal_Handler<T>::handleStatus, handler);
    sub->setQueueCapacity(1);
    while(0 == lcm_ptr.handle());
}

template <typename T>
__host__
void testMPC_LCM(lcm::LCM *lcm_ptr, trajVars<T> *tvars, algTrace<T> *atrace, matDimms *dimms, char hardware){
    // launch the simulator
    printf("Make sure the drake kuka simulator or kuka hardware is launched!!!\n");//, [F]ilter, [G]oal changer, and traj[R]unner are launched!!!\n");
	// get the max iters and time per solve
	int itersToDo = getMaxIters(1000, 1);
	int timeLimit = getTimeBudget(1000, 1); //note in ms
	// init the Ts
	tvars->t0_plant = 0; tvars->t0_sys = 0;	int64_t tActual_plant = 0; int64_t tActual_sys = 0;
    // allocate memory and construct the appropriate handlers and launch the threads
    std::thread mpcThread;                  //lcm::Subscription *mpcSub = nullptr;  // pass in sub objects so we can unsubscribe later
    CPUVars<T> *cvars = new CPUVars<T>;     LCM_MPCLoop_Handler<T> chandler = LCM_MPCLoop_Handler<T>(cvars,tvars,dimms,atrace,itersToDo,timeLimit);
    GPUVars<T> *gvars = new GPUVars<T>;     LCM_MPCLoop_Handler<T> ghandler = LCM_MPCLoop_Handler<T>(gvars,tvars,dimms,atrace,itersToDo,timeLimit);
    LCM_PickAndPlaceGoal_Handler<T> goalhandler = LCM_PickAndPlaceGoal_Handler<T>(0.001);
    if (hardware == 'G'){
		allocateMemory_GPU_MPC<T>(gvars, dimms, tvars);
		// load in inital trajectory and initial goal
		loadTraj<T>(tvars->x, tvars->u, tvars->KT, dimms->ld_x, dimms->ld_u, dimms->ld_KT);		goalhandler.loadInitialGoal(gvars->xGoal);
		for (int i = 0; i < NUM_ALPHA; i++){
			gpuErrchk(cudaMemcpy(gvars->h_d_x[i], tvars->x, (dimms->ld_x)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(gvars->h_d_u[i], tvars->u, (dimms->ld_u)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice));
		}
		memcpy(gvars->xActual, tvars->x, STATE_SIZE*sizeof(T));
		// note run to conversion with no time or iter limits
		runiLQR_MPC_GPU<T>(tvars,gvars,dimms,atrace,tActual_sys,tActual_plant,1);
		// then launch the MPC thread
     	mpcThread  = std::thread(&runMPCHandler<T>, lcm_ptr, &ghandler);    
     	// setCPUForThread(&mpcThread, 1);
    }
    else{
		allocateMemory_CPU_MPC<T>(cvars, dimms, tvars);
		// load in inital trajectory and goal
		loadTraj<T>(tvars->x, tvars->u, tvars->KT, dimms->ld_x, dimms->ld_u, dimms->ld_KT);		goalhandler.loadInitialGoal(cvars->xGoal);
		memcpy(cvars->x, tvars->x, (dimms->ld_x)*NUM_TIME_STEPS*sizeof(T));
		memcpy(cvars->u, tvars->u, (dimms->ld_u)*NUM_TIME_STEPS*sizeof(T));
		memcpy(cvars->xActual, tvars->x, STATE_SIZE*sizeof(T));
		// note run to conversion with no time or iter limits
		runiLQR_MPC_CPU<T>(tvars,cvars,dimms,atrace,tActual_sys,tActual_plant,1);
		// then launch the MPC thread
     	mpcThread = std::thread(&runMPCHandler<T>, lcm_ptr, &chandler);    
     	// setCPUForThread(&mpcThread, 1);
    }
    // launch the trajRunner
    std::thread trajThread = std::thread(&runTrajRunner<T>, dimms);
    // launch the goal monitor
    std::thread goalThread = std::thread(&runPickAndPlaceGoalLCM<algType>, &goalhandler);
    // launch the status filter if needed
    #if USE_VELOCITY_FILTER
    	std::thread filterThread = std::thread(&run_IIWA_STATUS_filter<algType>);
	#endif
    printf("All threads launched -- check simulator output!\n");
    mpcThread.join();	trajThread.join();	goalThread.join();
    #if USE_VELOCITY_FILTER
    	filterThread.join();
    #endif
    // printf("Threads Joined\n");
    if (hardware == 'G'){freeMemory_GPU_MPC<T>(gvars);}  else{freeMemory_CPU_MPC<T>(cvars);}     delete gvars;   delete cvars;
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	// test based on command line args
	char hardware = '?'; // require user input
	if (argc > 1){hardware = argv[1][0];}
	trajVars<algType> *tvars = new trajVars<algType>;	algTrace<algType> *atrace = new algTrace<algType>;	matDimms *dimms = new matDimms;
	if (hardware == 'C' || hardware == 'G'){
		lcm::LCM lcm_ptr;	if(!lcm_ptr.good()){printf("LCM Failed to Init\n"); return 1;}
		testMPC_LCM<algType>(&lcm_ptr,tvars,atrace,dimms,hardware);
	}
	// run the status filter
	else if (hardware == 'F'){
		run_IIWA_STATUS_filter<algType>();
	}
	// run the simulator
	else if (hardware == 'S'){
		double xInit[STATE_SIZE]; loadX<double>(xInit);
		runLCMSimulator(xInit);
	}
	// various printers
	else if (hardware == 'P'){
		char type = argv[1][1];
		lcm::LCM lcm_ptr;	if(!lcm_ptr.good()){printf("LCM Failed to Init\n"); return 1;} 
		if(type == 'S'){
			LCM_IIWA_STATUS_printer<algType> *shandler = new LCM_IIWA_STATUS_printer<algType>;
			run_IIWA_STATUS_printer<algType>(&lcm_ptr,shandler);
			delete shandler;	
		}
		else if(type == 'C'){
			LCM_IIWA_COMMAND_printer<algType> *chandler = new LCM_IIWA_COMMAND_printer<algType>;
			run_IIWA_COMMAND_printer<algType>(&lcm_ptr,chandler);
			delete chandler;
		}
		else if(type == 'T'){
			LCM_traj_printer<algType> *thandler = new LCM_traj_printer<algType>;
			run_traj_printer<algType>(&lcm_ptr,thandler);
			delete thandler;
		}
		else if(type == 'F'){
			LCM_IIWA_STATUS_FILTERED_printer<algType> *fhandler = new LCM_IIWA_STATUS_FILTERED_printer<algType>;
			run_IIWA_STATUS_FILTERED_printer<algType>(&lcm_ptr,fhandler);
			delete fhandler;
		}
		else{printf("Invalid printer requested as second char [%c]. Currently supports: [S]tatus, [C]ommand, [T]rajectory, or [F]iltered Status\n",type);}
	}
	// else error
	printf("Error: Unkown code - usage is: [C]PU or [G]PU MPC Algorithm, Debug [P]rinters, or Kuka [S]imulator\n"); hardware = '?';
	// free the trajVars and the wrappers
	freeTrajVars<algType>(tvars);	delete atrace;	delete tvars;	delete dimms;
	return (hardware == '?');
}
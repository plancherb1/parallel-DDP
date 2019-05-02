/***
nvcc -std=c++11 -o pickNPlace.exe LCM_pickNPlace_examples.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -llcm -gencode arch=compute_61,code=sm_61 -rdc=true -O3
***/
#define USE_WAFR_URDF 0
#define EE_COST 1
#define USE_SMOOTH_ABS 0
#define SMOOTH_ABS_ALPHA 0.2
// default cost terms for the start of the goal to drop the arm from 0 vector to the start of the fig 8
// delta xyz, delta rpy, u, xzyrpyd, xyzrpy
#define SMALL 0.00001
#define _Q_EE1 150.0
#define _Q_EE2 SMALL
#define _R_EE 0.001
#define _QF_EE1 150.0
#define _QF_EE2 SMALL
#define _Q_xdEE 10.0
#define _QF_xdEE 10.0
#define _Q_xEE SMALL
#define _QF_xEE SMALL

#define USE_EE_VEL_COST 0
#define _Q_EEV1 0.0
#define _Q_EEV2 0.0
#define _QF_EEV1 0.0
#define _QF_EEV2 0.0

#define USE_LIMITS_FLAG 0
#define R_TL 0.0
#define Q_PL 0.0
#define Q_VL 0.0

#define MPC_MODE 1
#define USE_LCM 1
#define USE_VELOCITY_FILTER 0
#define HARDWARE_MODE 1

#define IGNORE_MAX_ROX_EXIT 0
#define TOL_COST 0.00001
#define SOLVES_TO_RESET 15
#define PLANT 4
#define PI 3.14159

#define E_NORM_LIM 0.05
#define V_NORM_LIM 0.05

#include "../config.cuh"


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
int runMPC_LCM(char hardware, T *xInit){
    // launch the simulator
    printf("Make sure the drake kuka simulator or kuka hardware is launched!!!\n");//, [F]ilter, [G]oal changer, and traj[R]unner are launched!!!\n");
	// get the max iters and time per solve
	printf("What is the maximum number of iterations a solver can take? (q to exit)?\n");
	int itersToDo = getInt(1000, 1);
	printf("What should the MPC time budget be (in ms)? (q to exit)?\n");
	int timeLimit = getInt(1000, 1); //note in ms
	// allocate variables and load inital variables
	trajVars<T> *tvars = new trajVars<T>; matDimms *dimms = new matDimms; algTrace<T> *atrace = new algTrace<T>;
	costParams<T> *cst = new costParams<T>;	loadCost(cst); // load in default cost to start
    std::thread mpcThread; LCM_MPCLoop_Handler<T> *mpchandler; CPUVars<T> *cvars; GPUVars<T> *gvars; // pointers for reference later
    // spend time allocating for CPU / GPU split
    if (hardware == 'G'){gvars = new GPUVars<T>; allocateMemory_GPU_MPC<T>(gvars, dimms, tvars); loadTraj<T>(gvars, tvars, dimms, xInit);}
    else{				 cvars = new CPUVars<T>; allocateMemory_CPU_MPC<T>(cvars, dimms, tvars); loadTraj<T>(cvars, tvars, dimms, xInit);}
    // get the goal handler
    LCM_PickAndPlaceGoal_Handler<T> *goalhandler = new LCM_PickAndPlaceGoal_Handler<T>(E_NORM_LIM);
    // then load the goals and LCM handlers and launch the MPC threads
    if (hardware == 'G'){
    	// load initial goal and run to full convergence to warm start
    	goalhandler->loadInitialGoal(gvars->xGoal); runiLQR_MPC_GPU<T>(tvars,gvars,dimms,atrace,cst,0,0,1);
		// then create the handler and launch the MPC thread
		mpchandler = new LCM_MPCLoop_Handler<T>(gvars,tvars,dimms,atrace,cst,itersToDo,timeLimit);
     	mpcThread  = std::thread(&runMPCHandler<T>, mpchandler);    
    }
    else{
    	// load initial goal and run to full convergence to warm start
    	goalhandler->loadInitialGoal(cvars->xGoal); runiLQR_MPC_CPU<T>(tvars,cvars,dimms,atrace,cst,0,0,1);
		// then create the handler and launch the MPC thread
		mpchandler = new LCM_MPCLoop_Handler<T>(cvars,tvars,dimms,atrace,cst,itersToDo,timeLimit);
     	mpcThread  = std::thread(&runMPCHandler<T>, mpchandler);    
     	if(FORCE_CORE_SWITCHES){setCPUForThread(&mpcThread, 1);} // move to another CPU
    }
    // launch the goal monitor
    std::thread goalThread = std::thread(&runPickAndPlaceGoalLCM<T>, goalhandler);
    // launch the trajRunner
    std::thread trajThread = std::thread(&runTrajRunner<T>, dimms);
    // launch the status filter if needed
    #if USE_VELOCITY_FILTER
    	std::thread filterThread = std::thread(&run_IIWA_STATUS_filter<T>);
	#endif
    printf("All threads launched -- check simulator/hardware output!\n");
    mpcThread.join();	trajThread.join();	goalThread.join();
    #if USE_VELOCITY_FILTER
    	filterThread.join();
    #endif
    if (hardware == 'G'){freeMemory_GPU_MPC<T>(gvars); delete gvars;} else{freeMemory_CPU_MPC<T>(cvars); delete cvars;}
    freeTrajVars<T>(tvars); delete tvars; delete atrace; delete dimms; delete cst; delete mpchandler; delete goalhandler;
    return 0;
}

int main(int argc, char *argv[])
{
	// init rand
	srand(time(NULL));
	// initial state for example
	algType xInit[STATE_SIZE]; loadInitialState(xInit,1);
	// require user input for mode of operation
	char hardware = '?'; if (argc > 1){hardware = argv[1][0];}
	// run the MPC loop
	if (hardware == 'C' || hardware == 'G'){return runMPC_LCM<algType>(hardware,xInit);}
	// run aditional example options (printers, simulator, etc.)
	else{return runOtherOptions<algType>(argv,xInit);}
}
/***
nvcc -std=c++11 -o pickNPlace.exe LCM_pickNPlace_examples.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -llcm -gencode arch=compute_61,code=sm_61 -O3
***/
#define USE_WAFR_URDF 0
#define EE_COST 1
#define USE_SMOOTH_ABS 0
#define SMOOTH_ABS_ALPHA 0.2
// default cost terms for the start of the goal to drop the arm from 0 vector to the start of the fig 8
// delta xyz, delta rpy, u, xzyrpyd, xyzrpy
#define SMALL 0//0.00001
#define _Q_EE1 25.0
#define _Q_EE2 SMALL
#define _R_EE 0.001
#define _QF_EE1 250.0
#define _QF_EE2 SMALL
#define _Q_xdEE 10.0
#define _QF_xdEE 10.0
#define _Q_xEE SMALL
#define _QF_xEE SMALL
#define _Q_EE1_CLOSE 75.0
#define _QF_EE1_CLOSE 500.0
#define _Q_EE1_BIG 75.0
#define _QF_EE1_BIG 200.0

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

#define E_NORM_LIM 0.10
#define V_NORM_LIM 0.10
#define TRAJ_RUNNER_ALPHA 0 // smoothing on torque and pos commands per command [0-1] for percent new

#define TOTAL_TIME 1.0
#define NUM_TIME_STEPS 128

#include "../config.cuh"
#include <random>
std::default_random_engine randEng(time(0)); //seed
std::uniform_real_distribution<double> randX(0.4,0.6);
std::uniform_real_distribution<double> randY(0.35,0.75); // positive on one side negative on the other

template <typename T>
class LCM_PickAndPlaceGoal_Handler {
    public:
    	T eNormLim; T vNormLim; T curr_goal[6]; bool side; int iters; int time; T eNormMax; bool cstSent; bool closeCstSent; bool varsSent;
    	lcm::LCM lcm_ptr; // ptr to LCM object for publish ability
        // default cost function obj to update and send
        kuka::lcmt_cost_params defaultCst;

    	LCM_PickAndPlaceGoal_Handler(double eLim, double vLim, int _iters, int _time) : eNormLim(eLim), vNormLim(vLim), iters(_iters), time(_time) {
    		if(!lcm_ptr.good()){printf("LCM Failed to Init in Goal Handler\n");}
			side = 0; updateGoal(); curr_goal[2] = 0.1; // z is always 0.1 
            eNormMax = 0; cstSent = 1; closeCstSent = 0; varsSent = 1; // start with the standard cost and vars so don't need cstSent or eNormMax right now
    	}
    	~LCM_PickAndPlaceGoal_Handler(){}

    	// goal comp func
    	void updateGoal(){
    		curr_goal[0] = randX(randEng);
    		curr_goal[1] = randY(randEng) * (side ? -1 : 1);
    		side = !side;
    	}

    	// load initial goal
    	void loadInitialGoal(T *goal){for(int i = 0; i < 6; i++){goal[i] = curr_goal[i];}}

        // set default cost
        void setDefaultCost(const drake::lcmt_iiwa_status *msg){
            defaultCst.utime = msg->utime;  defaultCst.r_ee = _R_EE;       defaultCst.r = _R;
            defaultCst.q_ee1 = _Q_EE1;      defaultCst.q_ee2 = _Q_EE2;     defaultCst.qf_ee1 = _QF_EE1;   defaultCst.qf_ee2 = _QF_EE2;
            defaultCst.q_eev1 = _Q_EEV1;    defaultCst.q_eev2 = _Q_EEV2;   defaultCst.qf_eev1 = _QF_EEV1; defaultCst.qf_eev2 = _QF_EEV2;
            defaultCst.q_xdee = _Q_xdEE;    defaultCst.qf_xdee = _QF_xdEE; defaultCst.q_xee = _Q_xEE;     defaultCst.qf_xee = _QF_xEE;
            defaultCst.q1 = _Q1;            defaultCst.q2 = _Q2;           defaultCst.qf1 = _QF1;         defaultCst.qf2 = _QF2;
        }

		// update goal based on status
		void handleStatus(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
			// compute the e and v norms
			T eNorm; T vNorm; evNorm(msg,curr_goal,&eNorm,&vNorm);
            // debug print
            printf("[%f] eNorm[%f] vNorm[%f] for goal[%f %f %f]\n",static_cast<double>(msg->utime),eNorm,vNorm,curr_goal[0],curr_goal[1],curr_goal[2]);

            // then figure out if we are ready to change the goal
			if (eNorm < eNormLim && vNorm < vNormLim){
				updateGoal(); // first get a new goal and update the eNormMax
                evNorm(msg,curr_goal,&eNorm,&vNorm); eNormMax = eNorm;
				// then construct a goal message
				kuka::lcmt_target_twist dataOut;               					dataOut.utime = msg->utime;
				for (int i = 0; i < 3; i++){dataOut.position[i] = curr_goal[i];	dataOut.velocity[i] = 0;	
											dataOut.orientation[i] = 0;			dataOut.angular_velocity[i] = 0;}
				dataOut.orientation[3] = 0;
                // and construct a message to clearVars for the goal shift
                kuka::lcmt_solver_params dataOut2;  dataOut2.utime = msg->utime;
                dataOut2.timeLimit = time*10;       dataOut2.iterLimit = iters;        
                dataOut2.clearVars = 1;             dataOut2.useCostShift = 0;
                // // and send a larger Q/QF so that it can solve for a new traj
                setDefaultCost(msg); //defaultCst.q_xdee = 1.0; defaultCst.qf_xdee = 1.0; defaultCst.r_ee = 0.0001;
                // // reset publishing vars for next pass
                closeCstSent = 0; cstSent = 0; varsSent = 0;
                // and publish all of them
                lcm_ptr.publish(ARM_GOAL_CHANNEL,&dataOut);     lcm_ptr.publish(SOLVER_PARAMS_CHANNEL,&dataOut2);   lcm_ptr.publish(COST_PARAMS_CHANNEL,&defaultCst);
			}
            // if close do the close cost
            else if (!closeCstSent && eNorm < 2*eNormLim){
                setDefaultCost(msg);    defaultCst.q_ee1 = _Q_EE1_CLOSE;    defaultCst.qf_ee1 = _QF_EE1_CLOSE;
                closeCstSent = 1;       lcm_ptr.publish(COST_PARAMS_CHANNEL,&defaultCst);
            }
            // // as we start to move update the cost function so it isn't too aggressive again
            // else if (!cstSent && eNorm < 0.8*eNormMax){setDefaultCost(msg);    cstSent = 1;    lcm_ptr.publish(COST_PARAMS_CHANNEL,&defaultCst);}
            // // if just did a new traj solve then need to reset limits for MPC mode
            else if (!varsSent && eNorm < 0.95*eNormMax){
                kuka::lcmt_solver_params dataOut;   dataOut.utime = msg->utime;
                dataOut.timeLimit = time;           dataOut.iterLimit = iters;        dataOut.clearVars = 0;    dataOut.useCostShift = 1;
                varsSent = 1;                       lcm_ptr.publish(SOLVER_PARAMS_CHANNEL,&dataOut);
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
int runMPC_LCM(char mode, T *xInit){
    // launch the simulator
    printf("Make sure the drake kuka simulator or kuka hardware is launched!!!\n");//, [F]ilter, [G]oal changer, and traj[R]unner are launched!!!\n");
	// get the max iters and time per solve
	printf("What is the maximum number of iterations a solver can take? (q to exit)?\n");
	int itersToDo = getInt(1000, 1);
	printf("What should the MPC time budget be (in ms)? (q to exit)?\n");
	int timeLimit = getInt(1000, 1); //note in ms
    bool useCostShift = true;
	// allocate variables and load inital variables
	trajVars<T> *tvars = new trajVars<T>; matDimms *dimms = new matDimms; algTrace<T> *atrace = new algTrace<T>;
	costParams<T> *cst = new costParams<T>;	loadCost(cst); // load in default cost to start
    std::thread mpcThread; LCM_MPCLoop_Handler<T> *mpchandler; CPUVars<T> *cvars; GPUVars<T> *gvars; // pointers for reference later
    // spend time allocating for CPU / GPU split
    if (mode == 'G'){gvars = new GPUVars<T>; allocateMemory_GPU_MPC<T>(gvars, dimms, tvars); loadTraj<T>(gvars, tvars, dimms, xInit);}
    else{		     cvars = new CPUVars<T>; allocateMemory_CPU_MPC<T>(cvars, dimms, tvars); loadTraj<T>(cvars, tvars, dimms, xInit);}
    // get the goal handler
    LCM_PickAndPlaceGoal_Handler<T> *goalhandler = new LCM_PickAndPlaceGoal_Handler<T>(E_NORM_LIM,V_NORM_LIM,itersToDo,timeLimit);
    // then load the goals and LCM handlers and launch the MPC threads
    if (mode == 'G'){
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
		mpchandler = new LCM_MPCLoop_Handler<T>(cvars,tvars,dimms,atrace,cst,itersToDo,timeLimit,useCostShift);
     	mpcThread  = std::thread(&runMPCHandler<T>, mpchandler);    
     	if(FORCE_CORE_SWITCHES){setCPUForThread(&mpcThread, 1);} // move to another CPU
    }
    // launch the goal monitor
    std::thread goalThread = std::thread(&runPickAndPlaceGoalLCM<T>, goalhandler);
    // launch the trajRunner
    std::thread trajThread = std::thread(&runTrajRunner<T>, dimms, TRAJ_RUNNER_ALPHA);
    // launch the status filter if needed
    #if USE_VELOCITY_FILTER
    	std::thread filterThread = std::thread(&run_IIWA_STATUS_filter<T>);
	#endif
    printf("All threads launched -- check simulator/hardware output!\n");
    mpcThread.join();	trajThread.join();	goalThread.join();
    #if USE_VELOCITY_FILTER
    	filterThread.join();
    #endif
    if (mode == 'G'){freeMemory_GPU_MPC<T>(gvars); delete gvars;} else{freeMemory_CPU_MPC<T>(cvars); delete cvars;}
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
	char mode = '?'; if (argc > 1){mode = argv[1][0];}
	// run the MPC loop
	if (mode == 'C' || mode == 'G'){return runMPC_LCM<algType>(mode,xInit);}
	// run aditional example options (printers, simulator, etc.)
	else{return runOtherOptions<algType>(mode,xInit,argv);}
}
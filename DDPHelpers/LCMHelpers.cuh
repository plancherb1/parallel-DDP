/*****************************************************************
 * DDP LCM Helper Functions
 * (currently only supports iLQR - UDP in future release)
 *
 *   LCM_MPCLoop_Handler
 *   LCM_TrajRunner_Handler
 *   
 *
 *****************************************************************/
#include <lcm/lcm-cpp.hpp>
#include "../lcmtypes/drake/lcmt_iiwa_status.hpp"
#include "../lcmtypes/drake/lcmt_iiwa_command.hpp"
#include "../lcmtypes/drake/lcmt_trajectory_f.hpp"
#include "../lcmtypes/drake/lcmt_trajectory_d.hpp"
#include "../lcmtypes/kuka/lcmt_target_twist.hpp"
 #include "../lcmtypes/kuka/lcmt_target_position.hpp"
#include <type_traits>

const char *ARM_GOAL_CHANNEL    = "GOAL_CHANNEL";
const char *ARM_TRAJ_CHANNEL    = "TRAJ_CHANNEL";
const char *ARM_COMMAND_CHANNEL = "IIWA_COMMAND";
const char *ARM_STATUS_CHANNEL  = "IIWA_STATUS";
#if USE_VELOCITY_FILTER
    const char *ARM_STATUS_FILTERED = "IIWA_STATUS_FILTERED";
#else
    const char *ARM_STATUS_FILTERED = "IIWA_STATUS";
#endif
// #define GOAL_PUBLISHER_RATE_MS 30
// #define TEST_DELTA 0 // random small delta to keep things interesting (in ms) for tests

// intercept IIWA STATUS and compute velocity
template <typename T>
class LCM_IIWA_STATUS_filter {
    public:
        bool first_pass;    lcm::LCM lcm_ptr;
        int64_t prevTime;   double prevPos[NUM_POS];

        LCM_IIWA_STATUS_filter(){first_pass = 1;   if(!lcm_ptr.good()){printf("LCM Failed to Init in STATUS manager\n");}}
        ~LCM_IIWA_STATUS_filter(){}

        void run(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){                
            // first pass just publish the old msg
            if (first_pass || 1){
                lcm_ptr.publish(ARM_STATUS_FILTERED,msg); 
                // and set up for later passes
                first_pass = 0;     prevTime = msg->utime;
                for (int i = 0; i < NUM_POS; i++){prevPos[i] = msg->joint_position_measured[i];}
                return;
            }
            // otherwise build up a new message
            drake::lcmt_iiwa_status dataOut;                                
            dataOut.num_joints = msg->num_joints;                           dataOut.joint_position_measured.resize(dataOut.num_joints);     
            dataOut.joint_velocity_estimated.resize(dataOut.num_joints);    dataOut.joint_position_commanded.resize(dataOut.num_joints);    
            dataOut.joint_position_ipo.resize(dataOut.num_joints);          dataOut.joint_torque_measured.resize(dataOut.num_joints);
            dataOut.joint_torque_commanded.resize(dataOut.num_joints);      dataOut.joint_torque_external.resize(dataOut.num_joints);
            // copy out time, compute dt, and save time for next pass
            dataOut.utime = msg->utime;     double dt = static_cast<double>(msg->utime - prevTime)/1000000;     prevTime = msg->utime;
            for (int i = 0; i < NUM_POS; i++){
                // copy over pos and compute vel
                double pos = msg->joint_position_measured[i];
                dataOut.joint_position_measured[i] = pos;
                dataOut.joint_velocity_estimated[i] = (pos - prevPos[i])/dt;
                // save down pos for next time
                prevPos[i] = pos;
                // copy everything else
                dataOut.joint_position_commanded[i] = msg->joint_position_commanded[i];
                dataOut.joint_position_ipo[i] = msg->joint_position_ipo[i];
                dataOut.joint_torque_measured[i] = msg->joint_torque_measured[i];
                dataOut.joint_torque_commanded[i] = msg->joint_torque_commanded[i];
                dataOut.joint_torque_external[i] = msg->joint_torque_external[i];
            }    
            // publish out
            lcm_ptr.publish(ARM_STATUS_FILTERED,&dataOut);
        }
};

template <typename T>
__host__
void run_IIWA_STATUS_filter(){
    lcm::LCM lcm_ptr;   if(!lcm_ptr.good()){printf("LCM Failed to Init in status manager\n");}
    LCM_IIWA_STATUS_filter<T> *manager = new LCM_IIWA_STATUS_filter<T>;
    lcm::Subscription *sub = lcm_ptr.subscribe(ARM_STATUS_CHANNEL, &LCM_IIWA_STATUS_filter<T>::run, manager);
    sub->setQueueCapacity(1);
    while(0 == lcm_ptr.handle());
}

// trajRunner takes messages of new trajectories to execute and current status's and returns torque commands
template <typename T>
class LCM_TrajRunner {
    public:
        T *x, *u, *KT; // current trajectories
        int ld_x, ld_u, ld_KT; // dimms
        int64_t t0; // t0 for the current traj
        lcm::LCM lcm_ptr; // ptr to LCM object for publish ability
        int ready;

        // init local vars to match size of passed in vars and get LCM
        LCM_TrajRunner(int _ld_x, int _ld_u, int _ld_KT) : ld_x(_ld_x), ld_u(_ld_u), ld_KT(_ld_KT) {
            x = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));                 u = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
            KT = (T *)malloc(ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));      ready = 0;
            /*lcm_ptr = new lcm::LCM;*/  if(!lcm_ptr.good()){printf("LCM Failed to Init in Traj Runner\n");}
        } 
        // free and delete
        ~LCM_TrajRunner(){free(x); free(u); free(KT); /*delete lcm_ptr;*/}
        
        // lcm new traj callback function
        void newTrajCallback_f(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_trajectory_f *msg){
            memcpy(x, &(msg->x[0]), ld_x*NUM_TIME_STEPS*sizeof(float));
            memcpy(u, &(msg->u[0]), ld_u*NUM_TIME_STEPS*sizeof(float));
            memcpy(KT,&(msg->KT[0]),ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(float));
            t0 = msg->utime;
            ready = 1;
        }
        void newTrajCallback_d(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_trajectory_d *msg){
            memcpy(x, &(msg->x[0]), ld_x*NUM_TIME_STEPS*sizeof(double));
            memcpy(u, &(msg->u[0]), ld_u*NUM_TIME_STEPS*sizeof(double));
            memcpy(KT,&(msg->KT[0]),ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(double));
            t0 = msg->utime;
            ready = 1;
        }

        // lcm STATUS callback function
        void statusCallback(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){ 
            if(!ready){return;}
            // construct output msg container and begin to load it with data
            drake::lcmt_iiwa_command dataOut;   
            dataOut.num_joints = static_cast<int32_t>(NUM_POS);         dataOut.joint_position.resize(dataOut.num_joints);
            dataOut.utime = static_cast<int64_t>(msg->utime);           dataOut.joint_torque.resize(dataOut.num_joints);
            #pragma unroll
            for(int i=0; i < 6; i++){dataOut.wrench[i] = 0.0;}
            // get the correct controls for this time
            int err = getHardwareControls<T>(&(dataOut.joint_position[0]), &(dataOut.joint_torque[0]), 
                                             x, u, KT, static_cast<double>(t0), 
                                             &(msg->joint_position_measured[0]), &(msg->joint_velocity_estimated[0]),  
                                             static_cast<double>(msg->utime), ld_x, ld_u, ld_KT);
            // then publish
            if (!err){lcm_ptr.publish(ARM_COMMAND_CHANNEL,&dataOut);}
            else{printf("[!]CRITICAL ERROR: Asked to execute beyond bounds of current traj.\n");}
        }
};

template <typename T>
__host__
void runTrajRunner(matDimms *dimms){
    // init LCM and allocate a traj runner
    lcm::LCM lcm_ptr; if(!lcm_ptr.good()){printf("LCM Failed to Init in Traj Runner main loop\n");}
    LCM_TrajRunner<T> tr = LCM_TrajRunner<T>(dimms->ld_x, dimms->ld_u, dimms->ld_KT);
    // subscribe to everything
    lcm::Subscription *statusSub = lcm_ptr.subscribe(ARM_STATUS_FILTERED, &LCM_TrajRunner<T>::statusCallback, &tr);
    lcm::Subscription *trajSub;
    if (std::is_same<T, float>::value){
        trajSub = lcm_ptr.subscribe(ARM_TRAJ_CHANNEL, &LCM_TrajRunner<T>::newTrajCallback_f, &tr);
    }
    else{
        trajSub = lcm_ptr.subscribe(ARM_TRAJ_CHANNEL, &LCM_TrajRunner<T>::newTrajCallback_d, &tr);   
    }
    // only execute latest message (no lag)
    statusSub->setQueueCapacity(1); trajSub->setQueueCapacity(1);
    // handle forever
    while(0 == lcm_ptr.handle());
}

template <typename T>
class LCM_MPCLoop_Handler {
    public:
        GPUVars<T> *gvars; // local pointer to the global algorithm variables
        CPUVars<T> *cvars; // local pointer to the global algorithm variables
        matDimms *dimms; // pointer to mat dimms
        trajVars<T> *tvars; // local pointer to the global traj variables
        algTrace<T> *data; // local pointer to the global algorithm trace data stuff
        int iterLimit; int timeLimit; bool mode; // limits for solves and cpu/gpu mode
        lcm::LCM lcm_ptr; // ptr to LCM object for publish ability

        // init and store the global location
        LCM_MPCLoop_Handler(GPUVars<T> *avIn, trajVars<T> *tvarsIn, matDimms *dimmsIn, algTrace<T> *dataIn, int iL, int tL) : 
                            gvars(avIn), tvars(tvarsIn), dimms(dimmsIn), data(dataIn), iterLimit(iL), timeLimit(tL) {
                            /*lcm_ptr = new lcm::LCM;*/  if(!lcm_ptr.good()){printf("LCM Failed to Init in Traj Runner\n");}
                            cvars = nullptr; mode = 1;}
        LCM_MPCLoop_Handler(CPUVars<T> *avIn, trajVars<T> *tvarsIn, matDimms *dimmsIn, algTrace<T> *dataIn, int iL, int tL) : 
                            cvars(avIn), tvars(tvarsIn), dimms(dimmsIn), data(dataIn), iterLimit(iL), timeLimit(tL) {
                            /*lcm_ptr = new lcm::LCM;*/  if(!lcm_ptr.good()){printf("LCM Failed to Init in Traj Runner\n");}
                            gvars = nullptr; mode = 0;}
        ~LCM_MPCLoop_Handler(){/*delete lcm_ptr;*/} // do nothing in the destructor

        // lcm callback function for new arm goal (eePos)
        void handleGoalEE(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const kuka::lcmt_target_twist *msg){
            memcpy(gvars->xGoal,msg->position,3*sizeof(T));     memcpy(&(gvars->xGoal[3]),msg->velocity,3*sizeof(T));
        }
        // lcm callback function for new arm goal (q,qd)
        void handleGoalqqd(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const kuka::lcmt_target_position *msg){
            memcpy(gvars->xGoal,msg->position,NUM_POS*sizeof(T));   memcpy(&(gvars->xGoal[NUM_POS]),msg->velocity,NUM_POS*sizeof(T));
        }
    
        // lcm callback function for new arm status
        void handleStatus(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
            // determine if GPU or CPU mode and save down time and state pointers and try to get the 
            //    global lock and compute a new MPC iter if ready else return
            if(mode){if (!((gvars->lock)->try_lock())){return;}}             
               else {if (!((cvars->lock)->try_lock())){return;}}       
            int64_t tActual_plant = msg->utime;           
            struct timeval sys_t; gettimeofday(&sys_t,NULL);        int64_t tActual_sys = get_time_us_i64(sys_t);
            
            // if first load initialize timers and keep inital traj (so set xActual to x)
            T *xActual_load = mode == 1 ? gvars->xActual : cvars->xActual;
            if (tvars->first_pass){
                //printf("fist pass MPC loop\n");
                tvars->t0_plant = tActual_plant;    tvars->t0_sys = tActual_sys;     tvars->first_pass = false;
                #pragma unroll
                for (int i=0; i<STATE_SIZE; i++){xActual_load[i] = (T)(tvars->x)[i];}
            }
            // else update xActual
            else{
                #pragma unroll
                for (int i=0; i<NUM_POS; i++){xActual_load[i]         = (T)(msg->joint_position_measured)[i]; 
                                              xActual_load[i+NUM_POS] = (T)(msg->joint_velocity_estimated)[i];}
                if(mode){gpuErrchk(cudaMemcpy(gvars->d_xActual, gvars->xActual, STATE_SIZE*sizeof(T), cudaMemcpyHostToDevice));}
            }
            
            // run iLQR
            if(mode){runiLQR_MPC_GPU(tvars,gvars,dimms,data,tActual_sys,tActual_plant,0,iterLimit,timeLimit);  (gvars->lock)->unlock();}
            else{    runiLQR_MPC_CPU(tvars,cvars,dimms,data,tActual_sys,tActual_plant,0,iterLimit,timeLimit);  (cvars->lock)->unlock();}

            // publish to trajRunner
            if (std::is_same<T, float>::value){
                drake::lcmt_trajectory_f dataOut;               dataOut.utime = tvars->t0_plant;                int stepsSize = NUM_TIME_STEPS*sizeof(float);
                int xSize = (dimms->ld_x)*stepsSize;            int uSize = (dimms->ld_u)*stepsSize;            int KTSize = (dimms->ld_KT)*DIM_KT_c*stepsSize;
                dataOut.x_size = xSize;                         dataOut.u_size = uSize;                         dataOut.KT_size = KTSize;
                dataOut.x.resize(dataOut.x_size);               dataOut.u.resize(dataOut.u_size);               dataOut.KT.resize(dataOut.KT_size);
                memcpy(&(dataOut.x[0]),&(tvars->x[0]),xSize);   memcpy(&(dataOut.u[0]),&(tvars->u[0]),uSize);   memcpy(&(dataOut.KT[0]),&(tvars->KT[0]),KTSize);
                lcm_ptr.publish(ARM_TRAJ_CHANNEL,&dataOut);
            }
            else{
                drake::lcmt_trajectory_d dataOut;               dataOut.utime = tvars->t0_plant;                int stepsSize = NUM_TIME_STEPS*sizeof(double);   
                int xSize = (dimms->ld_x)*stepsSize;            int uSize = (dimms->ld_u)*stepsSize;            int KTSize = (dimms->ld_KT)*DIM_KT_c*stepsSize;
                dataOut.x_size = xSize;                         dataOut.u_size = uSize;                         dataOut.KT_size = KTSize;
                dataOut.x.resize(dataOut.x_size);               dataOut.u.resize(dataOut.u_size);               dataOut.KT.resize(dataOut.KT_size);
                memcpy(&(dataOut.x[0]),&(tvars->x[0]),xSize);   memcpy(&(dataOut.u[0]),&(tvars->u[0]),uSize);   memcpy(&(dataOut.KT[0]),&(tvars->KT[0]),KTSize);
                lcm_ptr.publish(ARM_TRAJ_CHANNEL,&dataOut);
            }
        }      
};

template <typename T>
__host__
void runMPCHandler(lcm::LCM *lcm_ptr, LCM_MPCLoop_Handler<T> *handler){
    lcm::Subscription *sub = lcm_ptr->subscribe(ARM_STATUS_FILTERED, &LCM_MPCLoop_Handler<T>::handleStatus, handler);
    #if defined EE_COST && EE_COST == 1
        lcm::Subscription *sub2 = lcm_ptr->subscribe(ARM_GOAL_CHANNEL, &LCM_MPCLoop_Handler<T>::handleGoalEE, handler);
    #else
        lcm::Subscription *sub2 = lcm_ptr->subscribe(ARM_GOAL_CHANNEL, &LCM_MPCLoop_Handler<T>::handleGoalqqd, handler);
    #endif
    sub->setQueueCapacity(1);   sub2->setQueueCapacity(1);
    while(0 == lcm_ptr->handle());
}

template <typename T>
class LCM_IIWA_STATUS_printer {
    public:
        LCM_IIWA_STATUS_printer(){}
        ~LCM_IIWA_STATUS_printer(){}

        void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){                
            // double eePos[NUM_POS];   compute_eePos_scratch<double>((double *)&(msg->joint_position_measured[0]), &eePos[0]);
            // printf("[%ld] eePos: [%f %f %f] w/ jointVel [%f %f %f %f %f %f %f]\n",msg->utime,eePos[0],eePos[1],eePos[2],
            //     msg->joint_velocity_estimated[0],msg->joint_velocity_estimated[1],msg->joint_velocity_estimated[2],
            //     msg->joint_velocity_estimated[3],msg->joint_velocity_estimated[4],msg->joint_velocity_estimated[5],
            //     msg->joint_velocity_estimated[6]);
            // printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
            //     msg->joint_position_measured[0],msg->joint_position_measured[1],msg->joint_position_measured[2],msg->joint_position_measured[3],
            //     msg->joint_position_measured[4],msg->joint_position_measured[5],msg->joint_position_measured[6],msg->joint_velocity_estimated[0],
            //     msg->joint_velocity_estimated[1],msg->joint_velocity_estimated[2],msg->joint_velocity_estimated[3],msg->joint_velocity_estimated[4],
            //     msg->joint_velocity_estimated[5],msg->joint_velocity_estimated[6]);
            printf("%ld | %f %f %f %f %f %f %f | %f %f %f %f %f %f %f | %f %f %f %f %f %f %f | %f %f %f %f %f %f %f\n",msg->utime,
                msg->joint_position_measured[0],msg->joint_position_measured[1],msg->joint_position_measured[2],msg->joint_position_measured[3],
                msg->joint_position_measured[4],msg->joint_position_measured[5],msg->joint_position_measured[6],
                msg->joint_velocity_estimated[0],msg->joint_velocity_estimated[1],msg->joint_velocity_estimated[2],msg->joint_velocity_estimated[3],
                msg->joint_velocity_estimated[4],msg->joint_velocity_estimated[5],msg->joint_velocity_estimated[6],
                msg->joint_torque_commanded[0],msg->joint_torque_commanded[1],msg->joint_torque_commanded[2],msg->joint_torque_commanded[3],
                msg->joint_torque_commanded[4],msg->joint_torque_commanded[5],msg->joint_torque_commanded[6],
                msg->joint_torque_measured[0],msg->joint_torque_measured[1],msg->joint_torque_measured[2],msg->joint_torque_measured[3],
                msg->joint_torque_measured[4],msg->joint_torque_measured[5],msg->joint_torque_measured[6]);
        }
};

template <typename T>
__host__
void run_IIWA_STATUS_printer(lcm::LCM *lcm_ptr, LCM_IIWA_STATUS_printer<T> *handler){
    lcm::Subscription *sub = lcm_ptr->subscribe(ARM_STATUS_CHANNEL, &LCM_IIWA_STATUS_printer<T>::handleMessage, handler);
    // sub->setQueueCapacity(1);
    while(0 == lcm_ptr->handle());
}

template <typename T>
class LCM_IIWA_STATUS_FILTERED_printer {
    public:
        LCM_IIWA_STATUS_FILTERED_printer(){}
        ~LCM_IIWA_STATUS_FILTERED_printer(){}

        void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){                
            double eePos[NUM_POS];   compute_eePos_scratch<double>((double *)&(msg->joint_position_measured[0]), &eePos[0]);
            printf("[%ld] eePos: [%f %f %f] w/ jointVel [%f %f %f %f %f %f %f]\n",msg->utime,eePos[0],eePos[1],eePos[2],
                msg->joint_velocity_estimated[0],msg->joint_velocity_estimated[1],msg->joint_velocity_estimated[2],
                msg->joint_velocity_estimated[3],msg->joint_velocity_estimated[4],msg->joint_velocity_estimated[5],
                msg->joint_velocity_estimated[6]);
        }
};

template <typename T>
__host__
void run_IIWA_STATUS_FILTERED_printer(lcm::LCM *lcm_ptr, LCM_IIWA_STATUS_FILTERED_printer<T> *handler){
    lcm::Subscription *sub = lcm_ptr->subscribe(ARM_STATUS_FILTERED, &LCM_IIWA_STATUS_FILTERED_printer<T>::handleMessage, handler);
    // sub->setQueueCapacity(1);
    while(0 == lcm_ptr->handle());
}

template <typename T>
class LCM_IIWA_COMMAND_printer {
    public:
        LCM_IIWA_COMMAND_printer(){}
        ~LCM_IIWA_COMMAND_printer(){}

        void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_command *msg){                
            printf("%ld %f %f %f %f %f %f %f\n",msg->utime,msg->joint_torque[0],msg->joint_torque[1],msg->joint_torque[2],
                                                 msg->joint_torque[3],msg->joint_torque[4],msg->joint_torque[5],msg->joint_torque[6]);
            // double eePos[NUM_POS];   compute_eePos_scratch<double>((double *)&(msg->joint_position[0]), &eePos[0]);
            // printf("[%ld] eePosDes: [%f %f %f] control: [%f %f %f %f %f %f %f ]\n",msg->utime,eePos[0],eePos[1],eePos[2],
            //                             msg->joint_torque[0],msg->joint_torque[1],msg->joint_torque[2],msg->joint_torque[3],
            //                             msg->joint_torque[4],msg->joint_torque[5],msg->joint_torque[6]);
        }
};

template <typename T>
__host__
void run_IIWA_COMMAND_printer(lcm::LCM *lcm_ptr, LCM_IIWA_COMMAND_printer<T> *handler){
    lcm::Subscription *sub = lcm_ptr->subscribe(ARM_COMMAND_CHANNEL, &LCM_IIWA_COMMAND_printer<T>::handleMessage, handler);
    // sub->setQueueCapacity(1);
    while(0 == lcm_ptr->handle());
}

template <typename T>
class LCM_traj_printer {
    public:
        LCM_traj_printer(){}
        ~LCM_traj_printer(){}

        void handleMessage_d(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_trajectory_d *msg){                
            double eePos[NUM_POS];   compute_eePos_scratch<double>((double *)&(msg->x[0]), &eePos[0]);
            printf("[%ld] new traj computed with eePos0: [%f %f %f]\n",msg->utime,eePos[0],eePos[1],eePos[2]);
        }
        void handleMessage_f(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_trajectory_f *msg){                
            float eePos[NUM_POS];   compute_eePos_scratch<float>((float *)&(msg->x[0]), &eePos[0]);
            printf("[%ld] new traj computed with eePos0: [%f %f %f]\n",msg->utime,eePos[0],eePos[1],eePos[2]);
        }
};

template <typename T>
__host__
void run_traj_printer(lcm::LCM *lcm_ptr, LCM_traj_printer<T> *handler){
    if (std::is_same<T, float>::value){
        lcm::Subscription *sub = lcm_ptr->subscribe(ARM_TRAJ_CHANNEL, &LCM_traj_printer<T>::handleMessage_f, handler);
    }
    else{
        lcm::Subscription *sub = lcm_ptr->subscribe(ARM_TRAJ_CHANNEL, &LCM_traj_printer<T>::handleMessage_d, handler);
    }
    // sub->setQueueCapacity(1);
    while(0 == lcm_ptr->handle());
}

class LCM_Simulator_Handler {
    public:
        int numSteps;               struct timeval start, end;   int64_t currTime;
        double nextX[STATE_SIZE];   double currX[STATE_SIZE];    double qdd[NUM_POS];
        double Tbody[36*NUM_POS];   double I[36*NUM_POS];        double torqueCom[CONTROL_SIZE]; 
        lcm::LCM lcm_ptr;           int hertz, debug;            double dt;

        LCM_Simulator_Handler(double *xInit, int _numSteps = 50, int _hertz = 1000, int _debug = 0) : numSteps(_numSteps), hertz(_hertz), debug(_debug) {
            for(int i=0; i < STATE_SIZE; i++){currX[i] = xInit[i];}
            for(int i=0; i < CONTROL_SIZE; i++){torqueCom[i] = 0;}
            if(!lcm_ptr.good()){printf("LCM Failed to Init in Simulator\n");}
            initI<double>(I);       initT<double>(Tbody);     gettimeofday(&start,NULL);    currTime = 0;       dt = 1.0/hertz;
        }
        ~LCM_Simulator_Handler(){}

        // lcm callback function to update the torqueCom
        void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_command *msg){
            #pragma unroll
            for(int i = 0; i < CONTROL_SIZE; i++){torqueCom[i] = msg->joint_torque[i];}
            // printf("currPosDes [%f %f %f %f %f %f %f] vs currPos [%f %f %f %f %f %f %f]\n",
            //     msg->joint_position[0],msg->joint_position[1],msg->joint_position[2],msg->joint_position[3],
            //     msg->joint_position[4],msg->joint_position[5],msg->joint_position[6],currX[0],currX[1],currX[2],currX[3],currX[4],currX[5],currX[6]);
        }

        // do simulation
        void simulate(double simTime){
            double prevX[STATE_SIZE];
            #pragma unroll
            for(int i = 0; i < STATE_SIZE; i++){prevX[i] = currX[i];}
            double currU[STATE_SIZE];
            #pragma unroll
            for(int i = 0; i < CONTROL_SIZE; i++){currU[i] = torqueCom[i];}

            currTime += static_cast<int64_t>(1000000*simTime);
            double stepTime = simTime/static_cast<double>(numSteps);
            for (int i=0; i< numSteps; i++){
                _integrator<double>(nextX,currX,currU,qdd,I,Tbody,stepTime);
                #pragma unroll
                for(int i = 0; i < STATE_SIZE; i++){currX[i] = nextX[i];}
            }
            if (debug == 1){
                printf("%f:%f %f %f %f %f %f %f:%f %f %f %f %f %f %f %f %f %f %f %f %f %f:%f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
                   simTime,
                   currU[0],currU[1],currU[2],currU[3],currU[4],currU[5],currU[6],
                   prevX[0],prevX[1],prevX[2],prevX[3],prevX[4],prevX[5],prevX[6],prevX[7],prevX[8],prevX[9],prevX[10],prevX[11],prevX[12],prevX[13],
                   currX[0],currX[1],currX[2],currX[3],currX[4],currX[5],currX[6],currX[7],currX[8],currX[9],currX[10],currX[11],currX[12],currX[13]);    
            }
            else if (debug == 2){
                double eePos[NUM_POS]; compute_eePos_scratch<double>(currX, &eePos[0]);
                printf("%f %f %f\n",eePos[0],eePos[1],eePos[2]); 
            }
            else if (debug == 3){
                printf("%f %f %f %f %f %f %f | %f %f %f %f %f %f %f\n",currX[0],currX[1],currX[2],currX[3],currX[4],currX[5],currX[6],currX[7],currX[8],currX[9],currX[10],currX[11],currX[12],currX[13]);
            }
        }

        // publish currX
        void publish(){
            //construct output msg container and begin to load it with data
            drake::lcmt_iiwa_status dataOut;                                dataOut.utime = currTime;
            dataOut.num_joints = static_cast<int32_t>(NUM_POS);             dataOut.joint_position_measured.resize(dataOut.num_joints);      
            dataOut.joint_velocity_estimated.resize(dataOut.num_joints);    dataOut.joint_position_commanded.resize(dataOut.num_joints);  
            dataOut.joint_position_ipo.resize(dataOut.num_joints);          dataOut.joint_torque_measured.resize(dataOut.num_joints);  
            dataOut.joint_torque_commanded.resize(dataOut.num_joints);      dataOut.joint_torque_external.resize(dataOut.num_joints);
            for(int i = 0; i < NUM_POS; i++){
                dataOut.joint_position_commanded[i] = 0;        dataOut.joint_position_ipo[i] = 0;
                double val = torqueCom[i];                      dataOut.joint_torque_external[i] = 0;
                dataOut.joint_torque_measured[i] = val;         dataOut.joint_torque_commanded[i] = val;
                dataOut.joint_position_measured[i] = currX[i];  dataOut.joint_velocity_estimated[i] = currX[i+NUM_POS];
            }
            lcm_ptr.publish(ARM_STATUS_CHANNEL,&dataOut);
        }

        // run the simulator for dt
        void runSim(){
            simulate(dt);
            double simTime;
            while(1){
                gettimeofday(&end,NULL);
                simTime = time_delta_s(start,end);
                if (simTime >= dt){
                    gettimeofday(&start,NULL);
                    publish();
                    break;
                }
            } 
        }
};


__host__
void runLCMSimulator(double *xInit, int numSteps = 50, int hertz = 1000, int debug = 0){
    lcm::LCM lcm_ptr;
    if(!lcm_ptr.good()){printf("LCM Failed to Init in Simulator\n");}
    LCM_Simulator_Handler handler = LCM_Simulator_Handler(xInit, numSteps, hertz, debug);
    lcm::Subscription *sub = lcm_ptr.subscribe(ARM_COMMAND_CHANNEL, &LCM_Simulator_Handler::handleMessage, &handler);
    sub->setQueueCapacity(1);
    // poll the fd for updates
    while(1){   
        int lcm_fd = lcm_ptr.getFileno();  fd_set fds;     FD_ZERO(&fds);  FD_SET(lcm_fd, &fds);
        struct timeval timeout = {0,10};   // seconds, microseconds to wait for message
        if (select(lcm_fd + 1, &fds, 0, 0, &timeout)) {if (FD_ISSET(lcm_fd, &fds)){lcm_ptr.handle();}} 
        handler.runSim();
    }
    // while(0 == lcm_ptr.handle());
}
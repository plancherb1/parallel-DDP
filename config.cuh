#ifndef _DDP_CONFIG_
#define _DDP_CONFIG_

/******************************************************************
 * Config.cuh
 *
 * User defined parameters for the MPC algorithms
 *
 * Also includes all helper files in order last due to dependencies
 *
 * Note: If you want to overide any defaults please set them in the
 *       example file before including this file
 *******************************************************************/

// load in utility functions and threading/cuda definitions
#include "utils/cudaUtils.h"
#include "utils/threadUtils.h"
#include <sys/time.h>

// options for examples depend on the plant
#ifndef PLANT
	#define PLANT 4 // 1:Pendulum 2:Cart_Pole 3:Quad 4:KukaArm
#endif
#if PLANT == 1
	#define NUM_POS 1
	#define STATE_SIZE (2*NUM_POS)
	#define CONTROL_SIZE 1
	#define RHO_INIT 10.0
#elif PLANT == 2
	#define NUM_POS 2
	#define STATE_SIZE (2*NUM_POS)
	#define CONTROL_SIZE 1
	#define MAX_DEFECT_SIZE 0.75
	#define RHO_INIT 10.0
#elif PLANT == 3
	#define NUM_POS 6
	#define STATE_SIZE (2*NUM_POS)
	#define CONTROL_SIZE 4
	#define ALPHA_BASE 0.5
	#define NUM_ALPHA 16
	#define MAX_DEFECT_SIZE 1.0
	#define RHO_INIT 1.0
#elif PLANT == 4
	#define NUM_POS 7
	#define STATE_SIZE (2*NUM_POS)
	#define CONTROL_SIZE 7
 	#define EE_TYPE 1 // flange but no EE on it
 	#ifndef TOTAL_TIME
		#define TOTAL_TIME 0.5
 	#endif
 	#ifndef NUM_TIME_STEPS
		#define NUM_TIME_STEPS 64
 	#endif
	#define ALPHA_BASE 0.5
	#define NUM_ALPHA 16
	#define MAX_DEFECT_SIZE 1.0
	#define RHO_INIT 12.5
	#define INTEGRATOR 1
#else
	#error "Currently only supports Simple Pendulum[1], Inverted Pendulum[2], Quadrotor [3], or KukaArm[4].\n"
#endif

// optiomizer options
#define DEBUG_SWITCH 0 // 1 for on 0 for off
#ifndef USE_ALG_TRACE
	#define USE_ALG_TRACE 1 // 1 for on 0 for off
#endif
#define USE_FINITE_DIFF 0 // 1 for on 0 for off (analytical vs. finite diff derivatives if needed)
#ifndef FINITE_DIFF_EPSILON
	#define FINITE_DIFF_EPSILON 0.00001
#endif
// define if we are working in doubles or floats
// typedef double algType;
typedef float algType;
// typedef half algType; // code needs reworking before this will work

// algorithmic options
#ifndef INTEGRATOR
	#define INTEGRATOR 3 // 1 for Euler, 2 for Midpoint, 3 for RK3
#endif
#define LINEAR_TRANSFORM_SWITCH 1 // 1 for on 0 for off
#define ALPHA_BEST_SWITCH 1 // 1 for on 0 for off
#define MAX_ITER 100
#define MAX_SOLVER_TIME 10000.0
#ifndef TOL_COST
	#define TOL_COST 0.0001 // % decrease
#endif

// parallelization options
#define M_BLOCKS 4
#define M_BLOCKS_B M_BLOCKS // how many time steps to do in parallel on back pass
#define M_BLOCKS_F M_BLOCKS // how many multiple shooting intervals to use in the forward pass
#define N_BLOCKS_B (NUM_TIME_STEPS/M_BLOCKS_B)
#define N_BLOCKS_F (NUM_TIME_STEPS/M_BLOCKS_F)
#define FORCE_PARALLEL 1 // 0 for better performance 1 for identical output for comp b/t CPU and GPU

// regularizer options
#define STATE_REG 1 // use Tassa state regularization (0 indicates standard Huu regularization)
#ifndef RHO_INIT
	#define RHO_INIT 1.0
#endif
#define RHO_MAX 10000000.0 //100000.0
#define RHO_MIN 0.01 // 0.000001
#define RHO_FACTOR 1.25
#ifndef IGNORE_MAX_ROX_EXIT
	#define IGNORE_MAX_ROX_EXIT 1
#endif

// line search options
#ifndef ALPHA_BASE
	#define ALPHA_BASE 0.75
#endif
#ifndef NUM_ALPHA
	#define NUM_ALPHA 32
#endif
#define USE_EXP_RED 1 // if 0 accept if dJ > 0 else use expected reduction formula (MIN < dJ/exp < MAX)
#ifndef EXP_RED_MIN
	#define EXP_RED_MIN 0.05 // use to not accept very small changes (set to 0 to ignore)
#endif
#ifndef EXP_RED_MAX
	#define EXP_RED_MAX 1.25 // use to not allow large bad jumps (set to \infty to ignore)
#endif
#define USE_MAX_DEFECT 1 // 0 to always accept if dJ/ExpRed is good and X will also limit to a reasonable defect per MAX_DEFECT_SIZE once a defect under X is acheived
#ifndef MAX_DEFECT_SIZE
	#define MAX_DEFECT_SIZE 1.0 // use to not allow new traj with non-physical jump artifacts in it from multiple shooting
#endif
#define onDefectBoundary(k) ((((k+1) % N_BLOCKS_F) == 0) && (k < NUM_TIME_STEPS - 1))

// task length / time
#ifndef TOTAL_TIME
	#define TOTAL_TIME 4.0
#endif
#ifndef NUM_TIME_STEPS
	#define NUM_TIME_STEPS 128
#endif
#define TIME_STEP (TOTAL_TIME/(NUM_TIME_STEPS-1))
#define get_time_us(time) (static_cast<double>(time.tv_sec * 1000000.0 + time.tv_usec))
#define get_time_ms(time) (get_time_us(time) / 1000.0)
#define time_delta_us(start,end) (static_cast<double>(get_time_us(end) - get_time_us(start)))
#define time_delta_ms(start,end) (time_delta_us(start,end)/1000.0)
#define time_delta_s(start,end) (time_delta_ms(start,end)/1000.0)

// GPU Stream Options
#define NUM_STREAMS (max(18,4+NUM_ALPHA))
// CPU Threading Options
#define USE_HYPER_THREADING 0 // assumes pairs are 0,CPU_CORES/2, etc. test with cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
#define FORCE_CORE_SWITCHES 0 // set to 1 to force a cycle across the cores (may improve speed b/c we know that tasks are independent and don't share cache in general)
#define CPU_CORES (std::thread::hardware_concurrency())
#if USE_HYPER_THREADING
	#define COST_THREADS (max(CPU_CORES,1))
	#define INTEGRATOR_THREADS (max(CPU_CORES,1))
	#define BP_THREADS (max(min(M_BLOCKS_B,2*CPU_CORES),1))
	#define FSIM_THREADS (max(min(M_BLOCKS_F,2*CPU_CORES),1))
	#define FSIM_ALPHA_THREADS (max(2*CPU_CORES/M_BLOCKS_F,1))
#else
	#define COST_THREADS (max(CPU_CORES/2,1))
	#define INTEGRATOR_THREADS (max(CPU_CORES/2,1))
	#define BP_THREADS (max(min(M_BLOCKS_B,CPU_CORES),1))
	#define FSIM_THREADS (max(min(M_BLOCKS_F,CPU_CORES),1))
	#define FSIM_ALPHA_THREADS (max(CPU_CORES/M_BLOCKS_F,1))
#endif
#define MAX_CPU_THREADS (max(max(13,max((max(FSIM_ALPHA_THREADS,FSIM_THREADS)+1)*COST_THREADS + INTEGRATOR_THREADS + 3, FSIM_ALPHA_THREADS*FSIM_ALPHA_THREADS + 1)),3*NUM_ALPHA+3))

// cost type options -- only applicable to Kuka arm (so default to 0)
#ifndef EE_COST
	#define EE_COST 0 // 1 for end effector based cost / goal and 0 for joint based
#endif
#ifndef USE_EE_VEL_COST
	#define USE_EE_VEL_COST 0 // turn on or off the eeVel code path
#endif
#ifndef USE_LIMITS_FLAG
	#define USE_LIMITS_FLAG 0 // use joint vel torque limits (quad pen)
#endif
#ifndef USE_SMOOTH_ABS
	#define USE_SMOOTH_ABS 0 // use smooth abs cost (only applicable to EE cost)
#endif
#ifndef CONSTRAINTS_ON
	#define CONSTRAINTS_ON 0 // AL style constraints
#endif

// dynamics URDF options (only applies to Kuka)
#ifndef USE_WAFR_URDF
	#define USE_WAFR_URDF 0
#endif
#ifndef MPC_MODE
	#define MPC_MODE 0 // sets gravity to 0
#endif

// MPC options
#ifndef USE_MAX_SOLVER_TIME
	#define USE_MAX_SOLVER_TIME 1
#endif

// Matrix Dimms
#define DIM_x_r STATE_SIZE
#define DIM_x_c 1
#define DIM_u_r CONTROL_SIZE
#define DIM_u_c 1
#define DIM_d_r STATE_SIZE
#define DIM_d_c 1
#define DIM_AB_r STATE_SIZE
#define DIM_AB_c (STATE_SIZE + CONTROL_SIZE)
#define DIM_ABT_r (STATE_SIZE + CONTROL_SIZE)
#define DIM_ABT_c STATE_SIZE
#define DIM_A_r STATE_SIZE
#define DIM_A_c STATE_SIZE
#define DIM_H_r (STATE_SIZE + CONTROL_SIZE)
#define DIM_H_c (STATE_SIZE + CONTROL_SIZE)
#define DIM_Hxx_r STATE_SIZE
#define DIM_Hxx_c STATE_SIZE
#define DIM_Hux_r CONTROL_SIZE
#define DIM_Hux_c STATE_SIZE
#define DIM_Hxu_r STATE_SIZE
#define DIM_Hxu_c CONTROL_SIZE
#define DIM_Huu_r CONTROL_SIZE
#define DIM_Huu_c CONTROL_SIZE
#define DIM_g_r (STATE_SIZE + CONTROL_SIZE)
#define DIM_g_c 1
#define DIM_gx_r STATE_SIZE
#define DIM_gx_c 1
#define DIM_gu_r CONTROL_SIZE
#define DIM_gu_c 1
#define DIM_P_r STATE_SIZE
#define DIM_P_c STATE_SIZE
#define DIM_p_r STATE_SIZE
#define DIM_p_c 1
#define DIM_K_r CONTROL_SIZE
#define DIM_K_c STATE_SIZE
#define DIM_KT_r DIM_K_c
#define DIM_KT_c DIM_K_r
#define DIM_du_r CONTROL_SIZE
#define DIM_du_c 1
#define OFFSET_HXU (DIM_x_r*(DIM_x_r+DIM_u_r))
#define OFFSET_HUU (OFFSET_HXU + DIM_x_r)
#define OFFSET_HUX_GU DIM_x_r
#define OFFSET_B (DIM_AB_r*DIM_x_r)
// Matrix Dimms

// include the correct set of cost and dynamics functions
#if PLANT == 1
	#include "plants/cost_pend.cuh"
	#include "plants/dynamics_pend.cuh"
#elif PLANT == 2
	#include "plants/cost_cart.cuh"
 	#include "plants/dynamics_cart.cuh"
#elif PLANT == 3
	#include "plants/cost_quad.cuh"
 	#include "plants/dynamics_quad.cuh"
#elif PLANT == 4
	#include "plants/cost_arm.cuh"
 	#include "plants/dynamics_arm.cuh"
#endif

// include integrators for those dynamics
#include "utils/integrators.cuh"

// 1: Backward Pass Helpers
#include "DDPHelpers/bpHelpers.cuh"

// 2: Forward Pass Helpers
#include "DDPHelpers/fpHelpers.cuh"

// 3: Next Iteration Setup and Init Helpers
#include "DDPHelpers/nisInitHelpers.cuh"

// 4: DDP Algorithm Wrappers
#if !defined(MPC_MODE) || MPC_MODE == 0
	#include "DDPHelpers/DDPWrappers.cuh"
#endif

// 5: MPC Helpers and Wrappers
#if defined(MPC_MODE) && MPC_MODE == 1
	#include "DDPHelpers/MPCHelpers.cuh"       
#endif

// 6: LCM Helpers and Wrappers
#if defined(USE_LCM) && USE_LCM == 1
	#include "DDPHelpers/LCMHelpers.cuh"
#endif

// finally include the example helpers
#include "utils/exampleUtils.cuh"

#endif
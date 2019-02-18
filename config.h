#ifndef _DDP_CONFIG_
#define _DDP_CONFIG_

/******************************************************************
 * User defined parameters
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
	#define RHO_INIT 1
#elif PLANT == 4
	#define NUM_POS 7
	#define STATE_SIZE (2*NUM_POS)
	#define CONTROL_SIZE 7
	#define TOTAL_TIME 0.5
	#define NUM_TIME_STEPS 64
	#define ALPHA_BASE 0.5
	#define NUM_ALPHA 16
	#define MAX_DEFECT_SIZE 1.0
	#define RHO_INIT 12.5
	#define INTEGRATOR 1
	#ifndef EE_COST
		#define EE_COST 0 // 1 for end effector based cost / goal and 0 for joint based
	#endif
	#define USE_LIMITS_FLAG 0 // use joint vel torque limits (quad pen)
#else
	#error "Currently only supports Simple Pendulum[1], Inverted Pendulum[2], Quadrotor [3], or KukaArm[4].\n"
#endif

// optiomizer options
#define DEBUG_SWITCH 1 // 1 for on 0 for off
#define USE_FINITE_DIFF 0 // 1 for on 0 for off (analytical vs. finite diff derivatives if needed)
#define FINITE_DIFF_EPSILON 0.00001
// define if we are working in doubles or floats
// typedef double algType;
typedef float algType;

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
#define M 1
#define M_B M // how many time steps to do in parallel on back pass
#define M_F M // how many multiple shooting intervals to use in the forward pass
#define N_B (NUM_TIME_STEPS/M_B)
#define N_F (NUM_TIME_STEPS/M_F)
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
#define onDefectBoundary(k) ((((k+1) % N_F) == 0) && (k < NUM_TIME_STEPS - 1))

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
#define USE_HYPER_THREADING 1 // assumes pairs are 0,CPU_CORES/2, etc. test with cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
#define FORCE_CORE_SWITCHES 0 // set to 1 to force a cycle across the cores (may improve speed b/c we know that tasks are independent and don't share cache in general)
#define CPU_CORES (std::thread::hardware_concurrency())
#if USE_HYPER_THREADING
	#define COST_THREADS (max(CPU_CORES,1))
	#define INTEGRATOR_THREADS (max(CPU_CORES,1))
	#define BP_THREADS (max(min(M_B,2*CPU_CORES),1))
	#define FSIM_THREADS (max(min(M_F,2*CPU_CORES),1))
	#define FSIM_ALPHA_THREADS (max(min(M_F,CPU_CORES),1))
#else
	#define COST_THREADS (max(CPU_CORES/2,1))
	#define INTEGRATOR_THREADS (max(CPU_CORES/2,1))
	#define BP_THREADS (max(min(M_B,CPU_CORES),1))
	#define FSIM_THREADS (max(min(M_F,2*CPU_CORES),1))
	#define FSIM_ALPHA_THREADS (max(min(M_F,CPU_CORES/2),1))
#endif
#define MAX_CPU_THREADS (max(max(13,max((max(FSIM_ALPHA_THREADS,FSIM_THREADS)+1)*COST_THREADS + INTEGRATOR_THREADS + 3, FSIM_ALPHA_THREADS*FSIM_ALPHA_THREADS + 1)),3*NUM_ALPHA+3))

// cost type options -- only applicable to Kuka arm (so default to 0)
#ifndef EE_COST
	#define EE_COST 0 // 1 for end effector based cost / goal and 0 for joint based
#endif
#ifndef USE_LIMITS_FLAG
	#define USE_LIMITS_FLAG 0 // use joint vel torque limits (quad pen)
#endif
#ifndef CONSTRAINTS_ON
	#define CONSTRAINTS_ON 0 // AL style constraints
#endif

// dynamics URDF options (only applies to Kuka)
#ifndef USE_WAFR_URDF
	#define USE_WAFR_URDF 0
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

#endif
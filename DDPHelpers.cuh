/*****************************************************************
 * DDP Helper Functions
 * (currently only supports iLQR - UDP in future release)
 *
 * Combines the following files (note they have cross dependencies 
 *                               which defines the order of imports):
 *   1: Backward Pass Helpers
 *   2: Forward Pass Helpers
 *   3: Next Iteration Setup and Init Helpers
 *	 4: DDP Algorithm Wrappers
 *   5: MPC Helpers
 *
 *****************************************************************/
// include util functions and config parameters file
#include "config.h"

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

// 5: MPC Helpers and Wrappers//
#if defined(MPC_MODE) && MPC_MODE == 1
	#include "DDPHelpers/MPCHelpers.cuh"       
#endif

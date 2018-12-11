#ifndef _THREAD_UTILS_
#define _THREAD_UTILS_

/*****************************************************************
 * CPP Thread Utils
 *
 * Launching struct and core switching code
 *****************************************************************/

#include <thread>
#include <pthread.h>

typedef struct _threadDesc_t {
   unsigned int tid;
   unsigned int dim;
   unsigned int reps;
} threadDesc_t;

#define compute_reps(tid,dim,dataN) (dataN/dim + static_cast<unsigned int>(tid < dataN%dim))

void setCPUForThread(std::thread thread[], int thread_id);

#endif
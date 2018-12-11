/*****************************************************************
 * CPP Thread Utils
 *
 * Launching struct and core switching code
 *****************************************************************/
#include <thread>
#include <pthread.h>

void setCPUForThread(std::thread thread[], int thread_id){
	 cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int CPU_id = (thread_id % std::thread::hardware_concurrency());
    CPU_SET(CPU_id,&cpuset);
    int rc = pthread_setaffinity_np(thread[thread_id].native_handle(),sizeof(cpu_set_t),&cpuset);
    if (rc != 0) {
      switch(rc){
      	case EINVAL: {printf("REQUESTED CPU[%d] UNAVAILABLE\n",CPU_id); break;}
      	case ESRCH: {printf("REQUESTED THREAD[%d] DOES NOT EXIST\n",thread_id); break;}
      	case EFAULT: {printf("BAD MEMORY ADDRESS FOR CPU SET\n"); break;}
      	default: {printf("UNKNOWN CPU AFFINITY ERROR -- POTENTIALLY PRIVILEDGE ISSUE\n");}
      }
    }
}
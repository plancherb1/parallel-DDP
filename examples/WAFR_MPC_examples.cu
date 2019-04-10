/***
nvcc -std=c++11 -o MPC.exe WAFR_MPC_examples.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -rdc=true -O3
***/
#define USE_WAFR_URDF 1
#define EE_COST 1

#define USE_SMOOTH_ABS 0
#define SMOOTH_ABS_ALPHA 0.2
#if USE_SMOOTH_ABS
	#define Q_HAND1 0.1		//2.0 // xyz
	#define Q_HAND2 0.001		//2.0 // rpy
	#define R_HAND 0.0001		//0.0001
	#define QF_HAND1 1000000.0	//20000.0 // xyz
	#define QF_HAND2 10000.0	//20000.0 // rpy
	#define Q_xdHAND 0.1		//1.0//0.1
	#define QF_xdHAND 10000.0	//10.0//100.0
	#define Q_xHAND 0.0		//0.0//0.001//1.0
	#define QF_xHAND 0.0		//0.0//1.0
#else
	#define Q_HAND1 0.1		//1.0 // xyz
	#define Q_HAND2 0//0.001		//1.0 // rpy
	#define R_HAND 0.0001		//0.001
	#define QF_HAND1 1000.0		//5000.0 // xyz
	#define QF_HAND2 0//10.0		//5000.0 // rpy
	#define Q_xdHAND 0.1		//1.0//0.1
	#define QF_xdHAND 1000.0	//10.0//100.0
	#define Q_xHAND 0.0		//0.0//0.001//1.0
	#define QF_xHAND 0.0		//0.0//1.0
#endif
#define Q_HANDV1 0
#define Q_HANDV2 0
#define QF_HANDV1 0
#define QF_HANDV2 0

#define MPC_MODE 1
#define USE_EE_VEL_COST 1
#define IGNORE_MAX_ROX_EXIT 0
#define TOL_COST 0.00001
#define PLANT 4
#include "../config.cuh"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

#define TEST_ITERS 1 // 100
#define ROLLOUT_FLAG 0
#define RANDOM_MEAN 0.0
#define RANDOM_STDEV 0.001
char errMsg[]  = "Error: Unkown code - usage is [C]PU or [G]PU with flag 1/0 for doFig8\n";
char tot[]  = "  TOT";		char init[] = " INIT";	char fsim[]   = "  SIM";	
char fsweep[]   = "SWEEP";	char bp[]   = "   BP";	char nis[]  = "  NIS";
std::default_random_engine randEng(time(0)); //seed
std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

#if PLANT == 1 // pend
	#error "MPC example defined for KukaArm[4].\n"
#elif PLANT == 2 // cart
	#error "MPC example defined for KukaArm[4].\n"
#elif PLANT == 3 // quad
 	#error "MPC example defined for KukaArm[4].\n"
#elif PLANT == 4 // arm
 	#if EE_COST
		#define PI 	   3.14159
		#define GOAL_X 0.3638
		#define GOAL_Y 0.0
		#define GOAL_Z 1.0628
		#define GOAL_r (0.5*PI)
		#define GOAL_p 0.0
		#define GOAL_y (0.5*PI)
	#else  
		#error "MPC example defined for KukaArm[4] with EE_COST.\n"
	#endif
#else
	#error "MPC example defined for KukaArm[4].\n"
#endif

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
int getTrajTime(int maxInt, int minInt){
   printf("How many seconds long should the tracked trajectory be? (q to exit)\n");
   return getInt(maxInt,minInt);
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
__host__
void printStats(std::vector<double> v, char *type){
	// sort gives us the median, max and min
    std::sort(v.begin(),v.end());
    int size = v.size();
	double _median = size % 2 ? v[size / 2] : (v[size / 2 - 1] + v[size / 2]) / 2.0;	
	double _max = v.back();		double _min = v.front();	double _stdev = 0.0;
	// sum gives us the average
	double sum = std::accumulate(v.begin(), v.end(), 0.0);	double _avg = sum / (double)size;	
	// and then the std dev	
	for(std::vector<double>::iterator it = v.begin(); it != v.end(); ++it){_stdev += pow(*it-_avg,2.0);}
	_stdev = pow(_stdev / (double)size, 0.5);
	printf("%s: Median[%f] Average[%f] StdDev[%f] max[%f] min[%f]\n",type,_median,_avg,_stdev,_max,_min);
}
template <typename T>
__host__ 
void printAllTimingStats(algTrace<T> *atrace){
	printStats(atrace->tTime,tot);
	printStats(atrace->initTime,init);
	printStats(atrace->simTime,fsim);
	printStats(atrace->sweepTime,fsweep);
	printStats(atrace->bpTime,bp);
	printStats(atrace->nisTime,nis);
}
template <typename T>
__host__ __forceinline__
void loadTraj(trajVars<T> *tvars, matDimms *dimms){
	T *xk = tvars->x;	T *uk = tvars->u;
	// for (int k=0; k<NUM_TIME_STEPS; k++){
	// 	xk[0] = (T)-0.5*PI;	xk[1] = (T)0.25*PI;	xk[2] = (T)0.167*PI;
	// 	xk[3] = (T)-0.167*PI;	xk[4] = (T)0.125*PI;	xk[5] = (T)0.167*PI;	xk[6] = (T)0.5*PI;
	// 	xk[7] = (T)randDist(randEng);	xk[8] = (T)randDist(randEng);	xk[9] = (T)randDist(randEng);
	// 	xk[10] = (T)randDist(randEng);	xk[11] = (T)randDist(randEng);	xk[12] = (T)randDist(randEng);	xk[13] = (T)randDist(randEng);
	// 	if (k < NUM_TIME_STEPS - 1){
	// 		uk[0] = 0.0;		uk[1] = -102.9832;	uk[2] = 11.1968;
	// 		uk[3] = 47.0724;	uk[4] = 2.5993;		uk[5] = -7.0290;	uk[6] = -0.0907;
	// 		//uk[0] = (T)0.01; uk[1] = (T)0.01; uk[2] = (T)0.01; uk[3] = (T)0.01; uk[4] = (T)0.01; uk[5] = (T)0.01; uk[6] = (T)0.01;
	// 	}
	// 	xk += (dimms->ld_x);	uk += (dimms->ld_u);
	// }
	for (int k=0; k<NUM_TIME_STEPS; k++){
		for (int i = 0; i < STATE_SIZE; i++){
			xk[i] = 0.0;	if (i < CONTROL_SIZE){uk[i] = 0.01;}
		}
		xk += (dimms->ld_x);	uk += (dimms->ld_u);
	}
	memset(tvars->KT, 0, (dimms->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
}
template <typename T>
__host__ __forceinline__
void loadGoal(CPUVars<T> *algvars){
	const T temp[] = {GOAL_X,GOAL_Y,GOAL_Z,GOAL_r,GOAL_p,GOAL_y};
	for (int i=0; i < 6; i++){(algvars->xGoal)[i] = temp[i];}
}
template <typename T>
__host__ __forceinline__
void loadGoal(GPUVars<T> *algvars){
	const T temp[] = {GOAL_X,GOAL_Y,GOAL_Z,GOAL_r,GOAL_p,GOAL_y};
	for (int i=0; i < 6; i++){(algvars->xGoal)[i] = temp[i];}
}
template <typename T>
__host__ __forceinline__
int loadGoal(T *goal, T *xGoals, T *yGoals, T *zGoals, double time, double tstep, double totalTime){
	int rep = 0;
	while(time > totalTime){time -= totalTime; rep++;}
	double delta = time/tstep;							double fraction = delta - std::floor(delta);
	int rd = static_cast<int>(std::floor(delta));		int ru = static_cast<int>(std::ceil(delta));
	goal[0] = (1-fraction)*xGoals[rd] + fraction*xGoals[ru];
	goal[1] = (1-fraction)*yGoals[rd] + fraction*yGoals[ru];
	goal[2] = (1-fraction)*zGoals[rd] + fraction*zGoals[ru];
	return rep;
}
template <typename T>
__host__ __forceinline__
int loadGoal(CPUVars<T> *algvars, T *xGoals, T *yGoals, T *zGoals, double time, double tstep, double totalTime){
	T goal[3];
	int rep = loadGoal(&goal[0],xGoals,yGoals,zGoals,time,tstep,totalTime);
	algvars->xGoal[0] = goal[0];	algvars->xGoal[1] = goal[1];	algvars->xGoal[2] = goal[2];
	algvars->xGoal[3] = 0;			algvars->xGoal[4] = 0;			algvars->xGoal[5] = 0;
	return rep;
}
template <typename T>
__host__ __forceinline__
int loadGoal(GPUVars<T> *algvars, T *xGoals, T *yGoals, T *zGoals, double time, double tstep, double totalTime){
	T goal[3];
	int rep = loadGoal(&goal[0],xGoals,yGoals,zGoals,time,tstep,totalTime);
	algvars->xGoal[0] = goal[0];	algvars->xGoal[1] = goal[1];	algvars->xGoal[2] = goal[2];
	algvars->xGoal[3] = 0;			algvars->xGoal[4] = 0;			algvars->xGoal[5] = 0;
	return rep;
}
template <typename T>
__host__ __forceinline__
int loadFig8Goal(T *goal, double time, double totalTime){
	T xGoals[] = {0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004};
	T yGoals[] = {0.13686922001827645,0.1281183229143938,0.11926059413541247,0.11030432312999372,0.1012577993467988,0.09212931223448896,0.08292715124172548,0.0736596058171696,0.06433496540948258,0.05496151946732569,0.045547557439360176,0.03610136877424733,0.026631242920648335,0.017145469327224508,0.007652337442637103,-0.001839863284452653,-0.011322843405383448,-0.020788313471494096,-0.030227984034123273,-0.039633565644609764,-0.0489967688542923,-0.05830930421450961,-0.06756288227660043,-0.07674921359190354,-0.08586000871175764,-0.09488697818749958,-0.1038655258329727,-0.11272375591745916,-0.12145267616480321,-0.13004109492853494,-0.13848752180117063,-0.1467858524541864,-0.1549281455978588,-0.1629165235905652,-0.1707482796892309,-0.17841599130217783,-0.18591882035832843,-0.1932502389635062,-0.20040076105913676,-0.20735798772338723,-0.21411519853223893,-0.22066551450768562,-0.2269966386554578,-0.2331053523296066,-0.23897989082424964,-0.24461177603116246,-0.24999586174091798,-0.255127737887603,-0.2600037985005273,-0.264623973362925,-0.2689800086072113,-0.27306787696210033,-0.2768840041676995,-0.2804279779836409,-0.28369462501981374,-0.28667669709392635,-0.2893761036490012,-0.29178950599107006,-0.29391158123626243,-0.29574034972478985,-0.29727453396452347,-0.29851358128156347,-0.2994544451742904,-0.30009482660988346,-0.30043907902783173,-0.30048644363405164,-0.3002369544043975,-0.29968819678814346,-0.2988444313574429,-0.2977048064508765,-0.2962719042333858,-0.29454317198576974,-0.2925200231035754,-0.2902045731500091,-0.2876022809123594,-0.2847133796914938,-0.28154140071122247,-0.27809048676073667,-0.27436270583704647,-0.2703665745543449,-0.26610524813698555,-0.26158238012185503,-0.25680474076422527,-0.2517742032662073,-0.2464935808285531,-0.24096598110942027,-0.23519855715870805,-0.22919065210989617,-0.22294815964121067,-0.216477551020404,-0.20978904844956445,-0.20289009717093992,-0.19578529751522875,-0.18849136560284127,-0.1810102720333381,-0.17335332623615154,-0.16552932116242403,-0.15754446533330008,-0.1494055335341655,-0.141119776212369,-0.13269158757842503,-0.12412621123921029,-0.11543344583116043,-0.10661783052843572,-0.09768974155699829,-0.08866263788534111,-0.07955032956578706,-0.07036309089673469,-0.06111613853088287,-0.051824258760106816,-0.04250384833178472,-0.03316965050170948,-0.02383739609051122,-0.014517898480146497,-0.00521285872839692,0.004084069854356758,0.013377716369566438,0.022678964415954628,0.03199791841264746,0.04134623887290685,0.05072829056150665,0.06014998301207159,0.06960936374549687,0.07908892160637689,0.08855068744478754,0.09794356384062872,0.10720878615596977,0.11629326145552335,0.12514991959047528,0.13374545516319478,0.14207020091055017,0.15012393938707272,0.15792746608076888,0.16551126865840357,0.1729042968130615,0.18013571994835326,0.1872239892183252,0.19418299168723718,0.20100573965434845,0.2076839316858779,0.2142059797503866,0.22054968628960595,0.22669845070103078,0.2326348093954518,0.2383522458039774,0.24384287073319427,0.2491022174907651,0.2541293098371375,0.2589202699947854,0.2634683865999179,0.2677700650071298,0.27181899023759926,0.2756002672767167,0.2791048721375289,0.28231577838682104,0.28522354073704403,0.2878230537497963,0.2901076694828819,0.2920783437903407,0.29374310403406084,0.295107993386028,0.2961895263765827,0.2970056784978085,0.2975819271104611,0.29794495005548816,0.298109264082986,0.2980740741579594,0.2978323971097816,0.2973591836734637,0.296622924641549,0.2955794967043364,0.29418154658089085,0.2923728011959445,0.2900962060295761,0.2873037962354698,0.28396753800504204,0.28032650994085717,0.2763886650341475,0.27216195627614015,0.26765433665806226,0.2628737591711411,0.2578281768066035,0.25252554255567694,0.2469738094095883,0.2411809303595648,0.2351548583968336,0.2289035465126218,0.22243494769815644,0.21575701494466482,0.208877701243374,0.201804959585511,0.19454674296230312,0.1871110043649775,0.179505696784761,0.1717387732128811,0.16381818664056474,0.1557518900590391,0.14754783645953137,0.1392139788332685,0.1307582701714778};
	T zGoals[] = {0.31756840547539744,0.32748579299353975,0.3378005717933035,0.3484792783644661,0.3594884491968049,0.3707946207800972,0.38236432960412026,0.39416411215865155,0.40616050493346834,0.4183200444183479,0.4306092671030677,0.44299470947740494,0.455442908031137,0.4679203992540413,0.48039371963589506,0.49282940566647565,0.5051939938355604,0.5174540206329267,0.5295760225483518,0.541526536071613,0.5532720976924878,0.5647792439007534,0.5760145111861872,0.5869444360385665,0.5975355549476685,0.607754404403259,0.617684862624283,0.6269833110602612,0.6356398015809976,0.6436439919363848,0.6509877754875409,0.6576613603932012,0.6636610682008525,0.6689826360732843,0.67361769235995,0.6775587102765505,0.6807894425694675,0.6832982853518634,0.6850685995152646,0.6860878343790213,0.6863475050397387,0.6858378423067206,0.6845542073433015,0.6824970237151333,0.679674950734998,0.6760903644476528,0.6717601780334646,0.6666986364356934,0.6609289904640975,0.6544774785385726,0.6473731907857357,0.6396420054814256,0.6313175949625012,0.6224353167682047,0.6130293234274724,0.6031408593626014,0.592806201725962,0.5820708170060365,0.5709740379170426,0.5595578699404182,0.5478698928627949,0.5359537973679934,0.5238581689279385,0.5116302430406027,0.4993208702348768,0.48697884889804044,0.4746532265736393,0.4623956012593628,0.4502485492310045,0.4382596910462214,0.42647866542842733,0.41494764320821875,0.4037154159777041,0.39282851027202587,0.3823335341683771,0.37227688282859595,0.36269856168881953,0.3536372194160615,0.3451304536909037,0.33721132075471055,0.32990808625336293,0.32324779496703726,0.3172587080115004,0.31196097533352374,0.3073723368592661,0.3035111825862362,0.3003931753833552,0.29803437903462704,0.2964420439874006,0.29562574916758594,0.2955914743255737,0.29633569503272444,0.29786307504133164,0.3001661781693752,0.30323465990169074,0.3070520940451594,0.3115998595664622,0.31685456748736623,0.3227967201975068,0.32939373258736826,0.336620401368088,0.34444137126548074,0.35282689981652365,0.3617486647489115,0.37117885795827055,0.3810878162585716,0.39144908831456365,0.4022280096944361,0.41338530657545247,0.42488278556704895,0.43667841400710394,0.4487240639652002,0.46096913859882693,0.47336419850958344,0.48584480962195276,0.4983474665337738,0.5108078008561817,0.5231640943169621,0.5353543092708679,0.5473260617454424,0.5590277695983947,0.5704169154454125,0.5814512174681081,0.5920942512854084,0.6023199392964617,0.6120999705542476,0.6214094905886763,0.6302189177557694,0.6384948605857313,0.6462095585403849,0.6533240617001391,0.6598152959296799,0.6656559786178493,0.6708180864798622,0.6752789417653723,0.6790134023420558,0.6819951301275096,0.6841972139799184,0.6856012933475145,0.6861863032005122,0.6859428344922742,0.6848659833744687,0.6829691340687087,0.6802671521438632,0.6767869940428872,0.6725559197263683,0.6675988289655357,0.6619506036376687,0.6556389703050904,0.6486908922059266,0.6411394935332501,0.6330153548211616,0.6243554553896911,0.6151955559582207,0.6055661079526033,0.5954998822170263,0.5850395750753018,0.5742180207932119,0.563083467349193,0.5516783030193103,0.5400466850891197,0.5282336225055959,0.5162846587691582,0.5042482411832561,0.49216898910506657,0.4800977462683095,0.4680759949262634,0.45614115835012375,0.4443280391401759,0.4326669709393109,0.42118572108087715,0.40991017010577024,0.3988536909103188,0.3880358943577357,0.37746325851094437,0.3677153816185399,0.3583626678734707,0.34943548878219216,0.34096421585115233,0.33297922058679913,0.3255108744955808,0.3185895490839454,0.31224561585834093,0.30650944632521565,0.30141141199101745,0.2969818843621946,0.29325123494519506,0.29024983524646697,0.28800805677245844,0.2865562710296175,0.28592484952439223,0.2861441637632308,0.28724458525258123,0.2892564854988917,0.2922102360086102,0.2961362082881848,0.30106477384406377,0.30702630418269494,0.3140511708105266,0.32216974523400665};
	int numGoals = 200; 	double tstep = totalTime/(numGoals-1);	double goalNum = time/tstep;
	double fraction = goalNum - std::floor(goalNum);				int rep = static_cast<int>(std::floor(goalNum)) / numGoals;
	int rd = static_cast<int>(std::floor(goalNum)) % numGoals;		int ru = static_cast<int>(std::ceil(goalNum)) % numGoals;
	goal[0] = (1-fraction)*xGoals[rd] + fraction*xGoals[ru];
	goal[1] = (1-fraction)*yGoals[rd] + fraction*yGoals[ru];
	goal[2] = (1-fraction)*zGoals[rd] + fraction*zGoals[ru];
	return rep;
}
template <typename T>
__host__ __forceinline__
int loadFig8Goal(CPUVars<T> *algvars, double time, double totalTime){
	T goal[3];		int rep = loadFig8Goal(&goal[0],time,totalTime);
	algvars->xGoal[0] = goal[0];	algvars->xGoal[1] = goal[1];	algvars->xGoal[2] = goal[2];
	algvars->xGoal[3] = 0;			algvars->xGoal[4] = 0;			algvars->xGoal[5] = 0;
	return rep;
}
template <typename T>
__host__ __forceinline__
int loadFig8Goal(GPUVars<T> *algvars, double time, double totalTime){
	T goal[3];		int rep = loadFig8Goal(&goal[0],time,totalTime);
	algvars->xGoal[0] = goal[0];	algvars->xGoal[1] = goal[1];	algvars->xGoal[2] = goal[2];
	algvars->xGoal[3] = 0;			algvars->xGoal[4] = 0;			algvars->xGoal[5] = 0;
	return rep;
}	
template <typename T>
__host__
void evNorm(T *xActual, T *xGoal, T *eNorm, T *vNorm){
	T eePos[NUM_POS];   compute_eePos_scratch<T>(xActual, &eePos[0]);
	*eNorm = static_cast<T>(sqrt(pow(eePos[0]-xGoal[0],2) + pow(eePos[1]-xGoal[1],2) + pow(eePos[2]-xGoal[2],2)));
	*vNorm = 0; for(int i=0;i<NUM_POS;i++){*vNorm+=(T)pow(xActual[NUM_POS+i],2);} *vNorm = static_cast<T>(sqrt(*vNorm));
}
template <typename T>
__host__
void evNorm(double *xActual, T *xGoal, T *eNorm, T *vNorm){
	T xActual2[STATE_SIZE];
	for(int i=0; i < STATE_SIZE; i++){xActual2[i] = static_cast<T>(xActual[i]);}
	evNorm<T>(xActual2,xGoal,eNorm,vNorm);
}
template <typename T, int SUBSTEPS>
__host__
T simulateForward(trajVars<T> *tvars, T *xActual, double elapsedTime, double goalTime, double totalTime){
	double I[36*NUM_POS]; double Tbody[36*NUM_POS]; initI<double>(I); initT<double>(Tbody);
	// and break elapsed time into SUBSTEPS for accuracy
	double qdes[NUM_POS], udes[CONTROL_SIZE], currX[STATE_SIZE], nextX[STATE_SIZE], qdd[NUM_POS];
	double dt_us = elapsedTime/static_cast<double>(SUBSTEPS); double dt = dt_us / 1000000.0;
	for(int j=0; j < STATE_SIZE; j++){currX[j] = static_cast<double>(xActual[j]);}
	double t0 = static_cast<double>(tvars->t0_plant); double tk = t0;
	T totalError = 0;	T goal[3];	T eNorm, vNorm;
	for(int i = 0; i < SUBSTEPS; i++){
		// get the goal and compute the error norm
		loadFig8Goal<T>(&goal[0],goalTime,totalTime);	evNorm<T>(currX,goal,&eNorm,&vNorm);	totalError += eNorm;
		// then get controls
		int err = getHardwareControls<T>(&(qdes[0]), 	&(udes[0]), 		tvars->x, 	tvars->u, 	tvars->KT, 	t0, 
                                         &(currX[0]),   &(currX[NUM_POS]),  tk, 
                                         tvars->ld_x, 	tvars->ld_u, 		tvars->ld_KT);
		if(err){printf("CRITICAL FAILURE ERROR ABORT MISSION\n");return 0;}
		// apply them
		_integrator<double>(&(nextX[0]),&(currX[0]),&(udes[0]),&(qdd[0]),&(I[0]),&(Tbody[0]),dt);
		for(int j=0; j < STATE_SIZE; j++){currX[j] = nextX[j];}		tk += dt_us;
	}
	#pragma unroll
	for(int j=0; j < STATE_SIZE; j++){xActual[j] = (T)currX[j];}
	loadFig8Goal<T>(&goal[0],goalTime,totalTime);	evNorm<T>(currX,goal,&eNorm,&vNorm);	totalError += eNorm;
	return (totalError / static_cast<T>(SUBSTEPS));
}
template <typename T>
__host__
void testMPC_lockstep(trajVars<T> *tvars, algTrace<T> *data, matDimms *dimms, char hardware, int doFig8){
	// define the requirements for "conversion" to the first goal
	T eNormLim = 0.05;	 T vNormLim = 0.05;	
	// define local variables
	double goalTime = 0; int initial_convergence_flag = 0; T eNorm, vNorm; T error = 0; int counter = 0; struct timeval start, end;
	// get the max iters per solve
	int itersToDo = getMaxIters(1000, 1);
	// get the max iters per solve
	int timeLimit = getTimeBudget(1000, 1); //note in ms
	// get the total time for the trajectory
	double totalTime_us = 1000000.0*static_cast<double>(getTrajTime(100, 1));	//double timePrint = 0;
	// init the Ts
	tvars->t0_plant = 0; tvars->t0_sys = 0;	int64_t tActual_plant = 0; int64_t tActual_sys = 0;
	if (hardware == 'G'){
		GPUVars<T> *algvars = new GPUVars<T>;
		allocateMemory_GPU_MPC<T>(algvars, dimms, tvars);
		// load in inital trajectory and goal
		loadTraj<T>(tvars, dimms);		loadFig8Goal<T>(algvars,goalTime,totalTime_us);
		for (int i = 0; i < NUM_ALPHA; i++){
			gpuErrchk(cudaMemcpy(algvars->h_d_x[i], tvars->x, (dimms->ld_x)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(algvars->h_d_u[i], tvars->u, (dimms->ld_u)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyHostToDevice));
		}
		memcpy(algvars->xActual, tvars->x, STATE_SIZE*sizeof(T));
		// note run to conversion with no time or iter limits
		runiLQR_MPC_GPU<T>(tvars,algvars,dimms,data,tActual_sys,tActual_plant,1);
		// then start a loop of run a couple steps simulate for X steps and repeat
		while(1){
			counter++;
			gettimeofday(&start,NULL);
			runiLQR_MPC_GPU<T>(tvars,algvars,dimms,data,tActual_sys,tActual_plant,0,itersToDo,timeLimit);
			gettimeofday(&end,NULL);
			double elapsedTime_us = time_delta_us(start,end); // TIME_STEP*1000000;//
			tvars->t0_plant = 0; 	tActual_plant = static_cast<int>(std::floor(elapsedTime_us));
			tvars->t0_sys = 0; 		tActual_sys = static_cast<int>(std::floor(elapsedTime_us));
      		error += simulateForward<T,150>(tvars,algvars->xActual,elapsedTime_us,goalTime,totalTime_us);
			// print where are we ending up and eePos
				int timeStepsTaken = static_cast<int>(elapsedTime_us/TIME_STEP/1000000);
				printf("[%d] With last successful [%d] ago\nSim of %.4f is %d steps goes to:\n",counter,tvars->last_successful_solve,elapsedTime_us,timeStepsTaken);
				printMat<T,1,STATE_SIZE>(algvars->xActual,1);
				printf(" With expected:\n");
				printMat<T,1,STATE_SIZE>(tvars->x + timeStepsTaken*(dimms->ld_x),1);
      		// print the state we sim to
				// T *xk = &(algvars->xActual[0]);
				// printf("%f,%f,%f,%f,%f,%f,%f,%f\n",timePrint,xk[0],xk[1],xk[2],xk[3],xk[4],xk[5],xk[6]);
				// timePrint += elapsedTime_us;
			// print the error and the end effector position
				T eePos[NUM_POS];   compute_eePos_scratch<T>(&(algvars->xActual[0]), &eePos[0]);
				evNorm(algvars->xActual, algvars->xGoal, &eNorm, &vNorm);
				printf("[[%f,%f,%f],[%f,%f,%f],%f,%f,%f],\n",eePos[0],eePos[1],eePos[2],algvars->xGoal[0],algvars->xGoal[1],algvars->xGoal[2],eNorm,error/counter,vNorm);
   			if (initial_convergence_flag){goalTime += elapsedTime_us;}
   			if(loadFig8Goal<T>(algvars,goalTime,totalTime_us) > 1){break;};
   			if (doFig8 && eNorm < eNormLim && vNorm < vNormLim){initial_convergence_flag = 1; error = 0; counter = 0;}
		}
		printf("\n\nAverage tracking error: [%f]\n",(error/counter));
		printAllTimingStats(data);
		freeMemory_GPU_MPC<T>(algvars);	delete algvars;
	}
	else{printf("CPU currently not implemented.\n");}
}
int main(int argc, char *argv[])
{
	srand(time(NULL));
	// test based on command line args
	char hardware = '?'; // require user input
	if (argc > 1){hardware = argv[1][0];}
	trajVars<algType> *tvars = new trajVars<algType>;	algTrace<algType> *atrace = new algTrace<algType>;	matDimms *dimms = new matDimms;
	if (hardware == 'C' || hardware == 'G'){
		int flag = atoi(&argv[1][1]);
		if (flag != 0 && flag != 1){printf("%s",errMsg); return 1;};
		testMPC_lockstep<algType>(tvars,atrace,dimms,hardware,flag);
	}
	else{printf("%s",errMsg); hardware = '?';}
	// free the trajVars and the wrappers
	freeTrajVars<algType>(tvars);	delete atrace;	delete tvars;	delete dimms;
	return (hardware == '?');
}
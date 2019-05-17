/***
nvcc -std=c++11 -o fig8.exe LCM_fig8_examples.cu ../utils/cudaUtils.cu ../utils/threadUtils.cpp -llcm -gencode arch=compute_61,code=sm_61 -O3
***/
#define USE_WAFR_URDF 0
#define EE_COST 1
#define USE_SMOOTH_ABS 0
#define USE_EE_VEL_COST 0
#define USE_LIMITS_FLAG 0

#define MPC_MODE 1
#define USE_LCM 1
#define USE_VELOCITY_FILTER 0
#define HARDWARE_MODE 1
#define USE_ALG_TRACE 0
#define USE_MAX_SOLVER_TIME 0
#define USE_FEEDBACK_IN_TRAJ_RUNNER 1
#define TRAJ_RUNNER_TIME_STEPS NUM_TIME_STEPS/4
#define PD_GAINS_ON_STATE 0

#define IGNORE_MAX_ROX_EXIT 0
#define TOL_COST 0.00001
#define SOLVES_TO_RESET 15
#define PLANT 4

#define E_NORM_LIM 0.05
#define V_NORM_LIM 0.05
#define TRAJ_RUNNER_ALPHA 0 // smoothing on torque and pos commands per command

#if USE_EE_VEL_COST
	// default cost terms for the start of the goal to drop the arm from the initial point to the start of the fig 8
	// delta xyz, delta rpy, u, xzyrpyd, xyzrpy
	#define SMALL 0//0.00001
	#define _Q_EE1 50.0
	#define _Q_EE2 SMALL
	#define _R_EE 0.001
	#define _QF_EE1 100.0
	#define _QF_EE2 SMALL
	#define _Q_xdEE 10.0
	#define _QF_xdEE 10.0
	#define _Q_xEE SMALL
	#define _QF_xEE SMALL
	#define _Q_EEV1 0.0
	#define _Q_EEV2 0.0
	#define _QF_EEV1 0.0
	#define _QF_EEV2 0.0
	// new cost terms for the actual fig 8 tracking
	#define _Q_EE1_fig8 300.0
	#define _Q_EE2_fig8 SMALL
	#define _R_EE_fig8 0.0005 // make 0.001 for the move to inital goal and then to 0.0005 for motion
	#define _QF_EE1_fig8 300.0
	#define _QF_EE2_fig8 SMALL
	#define _Q_xdEE_fig8 10.0
	#define _QF_xdEE_fig8 10.0
	#define _Q_xEE_fig8 1.0
	#define _QF_xEE_fig8 1.0
	#define _Q_EEV1_fig8 0
	#define _Q_EEV2_fig8 0
	#define _QF_EEV1_fig8 0
	#define _QF_EEV2_fig8 0
#else
	// default cost terms for the start of the goal to drop the arm from the initial point to the start of the fig 8
	// delta xyz, delta rpy, u, xzyrpyd, xyzrpy
	#define SMALL 0//0.00001
	#define _Q_EE1 50.0
	#define _Q_EE2 SMALL
	#define _R_EE 0.001
	#define _QF_EE1 100.0
	#define _QF_EE2 SMALL
	#define _Q_xdEE 10.0
	#define _QF_xdEE 10.0
	#define _Q_xEE SMALL
	#define _QF_xEE SMALL
	// new cost terms for the actual fig 8 tracking
	#define _Q_EE1_fig8 300.0
	#define _Q_EE2_fig8 SMALL
	#define _R_EE_fig8 0.001 // make 0.001 for the move to inital goal and then to 0.0005 for motion
	#define _QF_EE1_fig8 300.0
	#define _QF_EE2_fig8 SMALL
	#define _Q_xdEE_fig8 5.0
	#define _QF_xdEE_fig8 5.0
	#define _Q_xEE_fig8 5.0
	#define _QF_xEE_fig8 5.0
	#define _Q_EEV1_fig8 0
	#define _Q_EEV2_fig8 0
	#define _QF_EEV1_fig8 0
	#define _QF_EEV2_fig8 0
#endif

#include "../config.cuh"

template <typename T>
class LCM_Fig8Goal_Handler {
    public:
    	double totalTime;	double zeroTime;	int inFig8;
    	double eNormLim;	double vNormLim;	int costSent;
    	double totalError;	int numIters;		int currRep;
    	int iterLimit;		int timeLimit;
    	lcm::LCM lcm_ptr; // ptr to LCM object for publish ability
    	struct timeval start, end; int timeCount; double timeTotal;

    	LCM_Fig8Goal_Handler(double tTime, double eLim, double vLim, int iL, int tL) : 
    		totalTime(tTime), eNormLim(eLim), vNormLim(vLim), iterLimit(iL), timeLimit(tL) {
    		zeroTime = 0; inFig8 = 0; costSent = 0;	totalError = 0;	numIters = 0; currRep = 0;
    		if(!lcm_ptr.good()){printf("LCM Failed to Init in Goal Handler\n");}
    	}
    	~LCM_Fig8Goal_Handler(){}

    	// fig 8 goals
    	int loadFig8Goal(T *goal, double time){
			T xGoals[] = {0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004,0.6556285000000004};
			T yGoals[] = {0.13686922001827645,0.1281183229143938,0.11926059413541247,0.11030432312999372,0.1012577993467988,0.09212931223448896,0.08292715124172548,0.0736596058171696,0.06433496540948258,0.05496151946732569,0.045547557439360176,0.03610136877424733,0.026631242920648335,0.017145469327224508,0.007652337442637103,-0.001839863284452653,-0.011322843405383448,-0.020788313471494096,-0.030227984034123273,-0.039633565644609764,-0.0489967688542923,-0.05830930421450961,-0.06756288227660043,-0.07674921359190354,-0.08586000871175764,-0.09488697818749958,-0.1038655258329727,-0.11272375591745916,-0.12145267616480321,-0.13004109492853494,-0.13848752180117063,-0.1467858524541864,-0.1549281455978588,-0.1629165235905652,-0.1707482796892309,-0.17841599130217783,-0.18591882035832843,-0.1932502389635062,-0.20040076105913676,-0.20735798772338723,-0.21411519853223893,-0.22066551450768562,-0.2269966386554578,-0.2331053523296066,-0.23897989082424964,-0.24461177603116246,-0.24999586174091798,-0.255127737887603,-0.2600037985005273,-0.264623973362925,-0.2689800086072113,-0.27306787696210033,-0.2768840041676995,-0.2804279779836409,-0.28369462501981374,-0.28667669709392635,-0.2893761036490012,-0.29178950599107006,-0.29391158123626243,-0.29574034972478985,-0.29727453396452347,-0.29851358128156347,-0.2994544451742904,-0.30009482660988346,-0.30043907902783173,-0.30048644363405164,-0.3002369544043975,-0.29968819678814346,-0.2988444313574429,-0.2977048064508765,-0.2962719042333858,-0.29454317198576974,-0.2925200231035754,-0.2902045731500091,-0.2876022809123594,-0.2847133796914938,-0.28154140071122247,-0.27809048676073667,-0.27436270583704647,-0.2703665745543449,-0.26610524813698555,-0.26158238012185503,-0.25680474076422527,-0.2517742032662073,-0.2464935808285531,-0.24096598110942027,-0.23519855715870805,-0.22919065210989617,-0.22294815964121067,-0.216477551020404,-0.20978904844956445,-0.20289009717093992,-0.19578529751522875,-0.18849136560284127,-0.1810102720333381,-0.17335332623615154,-0.16552932116242403,-0.15754446533330008,-0.1494055335341655,-0.141119776212369,-0.13269158757842503,-0.12412621123921029,-0.11543344583116043,-0.10661783052843572,-0.09768974155699829,-0.08866263788534111,-0.07955032956578706,-0.07036309089673469,-0.06111613853088287,-0.051824258760106816,-0.04250384833178472,-0.03316965050170948,-0.02383739609051122,-0.014517898480146497,-0.00521285872839692,0.004084069854356758,0.013377716369566438,0.022678964415954628,0.03199791841264746,0.04134623887290685,0.05072829056150665,0.06014998301207159,0.06960936374549687,0.07908892160637689,0.08855068744478754,0.09794356384062872,0.10720878615596977,0.11629326145552335,0.12514991959047528,0.13374545516319478,0.14207020091055017,0.15012393938707272,0.15792746608076888,0.16551126865840357,0.1729042968130615,0.18013571994835326,0.1872239892183252,0.19418299168723718,0.20100573965434845,0.2076839316858779,0.2142059797503866,0.22054968628960595,0.22669845070103078,0.2326348093954518,0.2383522458039774,0.24384287073319427,0.2491022174907651,0.2541293098371375,0.2589202699947854,0.2634683865999179,0.2677700650071298,0.27181899023759926,0.2756002672767167,0.2791048721375289,0.28231577838682104,0.28522354073704403,0.2878230537497963,0.2901076694828819,0.2920783437903407,0.29374310403406084,0.295107993386028,0.2961895263765827,0.2970056784978085,0.2975819271104611,0.29794495005548816,0.298109264082986,0.2980740741579594,0.2978323971097816,0.2973591836734637,0.296622924641549,0.2955794967043364,0.29418154658089085,0.2923728011959445,0.2900962060295761,0.2873037962354698,0.28396753800504204,0.28032650994085717,0.2763886650341475,0.27216195627614015,0.26765433665806226,0.2628737591711411,0.2578281768066035,0.25252554255567694,0.2469738094095883,0.2411809303595648,0.2351548583968336,0.2289035465126218,0.22243494769815644,0.21575701494466482,0.208877701243374,0.201804959585511,0.19454674296230312,0.1871110043649775,0.179505696784761,0.1717387732128811,0.16381818664056474,0.1557518900590391,0.14754783645953137,0.1392139788332685,0.1307582701714778};
			T zGoals[] = {0.31756840547539744,0.32748579299353975,0.3378005717933035,0.3484792783644661,0.3594884491968049,0.3707946207800972,0.38236432960412026,0.39416411215865155,0.40616050493346834,0.4183200444183479,0.4306092671030677,0.44299470947740494,0.455442908031137,0.4679203992540413,0.48039371963589506,0.49282940566647565,0.5051939938355604,0.5174540206329267,0.5295760225483518,0.541526536071613,0.5532720976924878,0.5647792439007534,0.5760145111861872,0.5869444360385665,0.5975355549476685,0.607754404403259,0.617684862624283,0.6269833110602612,0.6356398015809976,0.6436439919363848,0.6509877754875409,0.6576613603932012,0.6636610682008525,0.6689826360732843,0.67361769235995,0.6775587102765505,0.6807894425694675,0.6832982853518634,0.6850685995152646,0.6860878343790213,0.6863475050397387,0.6858378423067206,0.6845542073433015,0.6824970237151333,0.679674950734998,0.6760903644476528,0.6717601780334646,0.6666986364356934,0.6609289904640975,0.6544774785385726,0.6473731907857357,0.6396420054814256,0.6313175949625012,0.6224353167682047,0.6130293234274724,0.6031408593626014,0.592806201725962,0.5820708170060365,0.5709740379170426,0.5595578699404182,0.5478698928627949,0.5359537973679934,0.5238581689279385,0.5116302430406027,0.4993208702348768,0.48697884889804044,0.4746532265736393,0.4623956012593628,0.4502485492310045,0.4382596910462214,0.42647866542842733,0.41494764320821875,0.4037154159777041,0.39282851027202587,0.3823335341683771,0.37227688282859595,0.36269856168881953,0.3536372194160615,0.3451304536909037,0.33721132075471055,0.32990808625336293,0.32324779496703726,0.3172587080115004,0.31196097533352374,0.3073723368592661,0.3035111825862362,0.3003931753833552,0.29803437903462704,0.2964420439874006,0.29562574916758594,0.2955914743255737,0.29633569503272444,0.29786307504133164,0.3001661781693752,0.30323465990169074,0.3070520940451594,0.3115998595664622,0.31685456748736623,0.3227967201975068,0.32939373258736826,0.336620401368088,0.34444137126548074,0.35282689981652365,0.3617486647489115,0.37117885795827055,0.3810878162585716,0.39144908831456365,0.4022280096944361,0.41338530657545247,0.42488278556704895,0.43667841400710394,0.4487240639652002,0.46096913859882693,0.47336419850958344,0.48584480962195276,0.4983474665337738,0.5108078008561817,0.5231640943169621,0.5353543092708679,0.5473260617454424,0.5590277695983947,0.5704169154454125,0.5814512174681081,0.5920942512854084,0.6023199392964617,0.6120999705542476,0.6214094905886763,0.6302189177557694,0.6384948605857313,0.6462095585403849,0.6533240617001391,0.6598152959296799,0.6656559786178493,0.6708180864798622,0.6752789417653723,0.6790134023420558,0.6819951301275096,0.6841972139799184,0.6856012933475145,0.6861863032005122,0.6859428344922742,0.6848659833744687,0.6829691340687087,0.6802671521438632,0.6767869940428872,0.6725559197263683,0.6675988289655357,0.6619506036376687,0.6556389703050904,0.6486908922059266,0.6411394935332501,0.6330153548211616,0.6243554553896911,0.6151955559582207,0.6055661079526033,0.5954998822170263,0.5850395750753018,0.5742180207932119,0.563083467349193,0.5516783030193103,0.5400466850891197,0.5282336225055959,0.5162846587691582,0.5042482411832561,0.49216898910506657,0.4800977462683095,0.4680759949262634,0.45614115835012375,0.4443280391401759,0.4326669709393109,0.42118572108087715,0.40991017010577024,0.3988536909103188,0.3880358943577357,0.37746325851094437,0.3677153816185399,0.3583626678734707,0.34943548878219216,0.34096421585115233,0.33297922058679913,0.3255108744955808,0.3185895490839454,0.31224561585834093,0.30650944632521565,0.30141141199101745,0.2969818843621946,0.29325123494519506,0.29024983524646697,0.28800805677245844,0.2865562710296175,0.28592484952439223,0.2861441637632308,0.28724458525258123,0.2892564854988917,0.2922102360086102,0.2961362082881848,0.30106477384406377,0.30702630418269494,0.3140511708105266,0.32216974523400665};
			int numGoals = 200; 	double tstep = totalTime/(numGoals-1);	double goalNum = time/tstep;
			double fraction = goalNum - std::floor(goalNum);				int rep = static_cast<int>(std::floor(goalNum)) / numGoals;
			int rd = static_cast<int>(std::floor(goalNum)) % numGoals;		int ru = static_cast<int>(std::ceil(goalNum)) % numGoals;
			goal[0] = (1-fraction)*xGoals[rd] + fraction*xGoals[ru];		goal[3] = 0.0;
			goal[1] = (1-fraction)*yGoals[rd] + fraction*yGoals[ru];		goal[4] = 0.0;
			goal[2] = (1-fraction)*zGoals[rd] + fraction*zGoals[ru];		goal[5] = 0.0;
			// goal[1] = goal[1] * 1.75;
			// goal[2] = goal[2] * 1.25;
			return rep;
		}

		// load initial goal
    	void loadInitialGoal(T *goal){loadFig8Goal(goal,0);}

    	// load nominal target
    	void loadInitialTarget(T *goal, T *target = nullptr){for(int i = 0; i < STATE_SIZE; i++){goal[i] = (target == nullptr) ? 0 : target[i];}}

    	// keep track of traj times
    	void newTrajCallback_f(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_trajectory_f *msg){
            if (inFig8){gettimeofday(&end,NULL); timeCount++; timeTotal += time_delta_ms(start,end);} gettimeofday(&start,NULL);
        }
        void newTrajCallback_d(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_trajectory_d *msg){
            if (inFig8){gettimeofday(&end,NULL); timeCount++; timeTotal += time_delta_ms(start,end);} gettimeofday(&start,NULL);
        }

		// update goal based on status
		void handleStatus(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
			// get current goal
			T goal[3]; double time = inFig8 ? msg->utime - zeroTime : 0; int rep = loadFig8Goal(goal,time);
			// compute the position error norm and velocity norm
			T eNorm; T vNorm; T currX[STATE_SIZE]; T eePos[NUM_POS];
			for(int i=0; i < STATE_SIZE; i++){
				if(i < NUM_POS){currX[i] = static_cast<T>(msg->joint_position_measured[i]);}
				else{			currX[i] = static_cast<T>(msg->joint_velocity_estimated[i-NUM_POS]);}
			}
			evNorm<T>(currX, goal, &eNorm, &vNorm, eePos);		totalError += static_cast<double>(eNorm);	numIters++;
			// debug print
			// printf("[%f] eNorm[%f] vNorm[%f] for goal[%f %f %f] and Pos[%f %f %f]\n",static_cast<double>(msg->utime),eNorm,vNorm,goal[0],goal[1],goal[2],eePos[0],eePos[1],eePos[2]);
			// print the error for each rep
			if(rep > currRep){
				printf("[!] Rep [%d] has total error [%f] with time [%f]\n",rep,totalError/numIters,timeTotal/timeCount); 
				totalError = 0; numIters = 0; currRep++; timeCount = 0; timeTotal = 0;
			}
			// then figure out if we are in the goal moving time
			if(inFig8){
				// then load in goal pos and zero out vel, orientation, angularVelocity (for now) -- note orientation is size 4 (quat)
				kuka::lcmt_target_twist dataOut;               dataOut.utime = msg->utime;
				for (int i = 0; i < 3; i++){dataOut.position[i] = goal[i];	dataOut.velocity[i] = 0;	
											dataOut.orientation[i] = 0;		dataOut.angular_velocity[i] = 0;}
				dataOut.orientation[3] = 0;
				// and publish it to goal channel
			    lcm_ptr.publish(ARM_GOAL_CHANNEL,&dataOut);
			}
			else {
				// else check to see if we should update goal next time
				if (eNorm < eNormLim && vNorm < vNormLim){
					// reset the zeroTime and set that we are inFig8
					zeroTime = msg->utime;		inFig8 = 1;		totalError = 0;		numIters = 0;
					// also update the solver params for this experiment
					kuka::lcmt_solver_params dataOut;	dataOut.utime = msg->utime;
					dataOut.timeLimit = timeLimit;		dataOut.iterLimit = iterLimit;		
					dataOut.clearVars = 0;              dataOut.useCostShift = 0;
					lcm_ptr.publish(SOLVER_PARAMS_CHANNEL,&dataOut);
				}
				// else if close but not there yet update the cost func to care more about moving to goals
				else if (!costSent && eNorm < 2.5*eNormLim && vNorm < 2.5*vNormLim){
					kuka::lcmt_cost_params dataOut;		dataOut.utime = msg->utime;
					dataOut.q_ee1 = _Q_EE1_fig8;		dataOut.q_ee2 = _Q_EE2_fig8;
					dataOut.qf_ee1 = _QF_EE1_fig8;		dataOut.qf_ee2 = _QF_EE2_fig8;
					dataOut.q_eev1 = _Q_EEV1_fig8;		dataOut.q_eev2 = _Q_EEV2_fig8;
					dataOut.qf_eev1 = _QF_EEV1_fig8;	dataOut.qf_eev2 = _QF_EEV2_fig8;
					dataOut.q_xdee = _Q_xdEE_fig8;		dataOut.qf_xdee = _QF_xdEE_fig8;
					dataOut.q_xee = _Q_xEE_fig8;		dataOut.qf_xee = _QF_xEE_fig8;
					dataOut.r_ee = _R_EE_fig8;			dataOut.r = _R;
					dataOut.q1 = _Q1; 					dataOut.q2 = _Q2;
					dataOut.qf1 = _QF1; 				dataOut.qf2 = _QF2;
					lcm_ptr.publish(COST_PARAMS_CHANNEL,&dataOut);
					costSent = 1;
				}
			}
			
		}
};
template <typename T>
void runFig8GoalLCM(LCM_Fig8Goal_Handler<T> *handler){
	lcm::LCM lcm_ptr; if(!lcm_ptr.good()){printf("LCM Failed to init in goal handler\n");}
	lcm::Subscription *sub = lcm_ptr.subscribe(ARM_STATUS_FILTERED, &LCM_Fig8Goal_Handler<T>::handleStatus, handler); lcm::Subscription *sub2;
	if (std::is_same<T, float>::value){sub2 = lcm_ptr.subscribe(ARM_TRAJ_CHANNEL, &LCM_Fig8Goal_Handler<T>::newTrajCallback_f, handler);}
    else if (std::is_same<T, double>::value){sub2 = lcm_ptr.subscribe(ARM_TRAJ_CHANNEL, &LCM_Fig8Goal_Handler<T>::newTrajCallback_d, handler);}
    else{printf("Timing only defined for floats and doubles\n");}
    sub->setQueueCapacity(1); sub2->setQueueCapacity(1);
    while(0 == lcm_ptr.handle());
    // while(1){lcm_ptr.handle();usleep(5000);}
}

template <typename T>
__host__
int runMPC_LCM(char mode, T *xInit){
	// launch the simulator
    // printf("Make sure the drake kuka simulator or kuka hardware is launched!!!\n");
	// get the max iters and time per solve
	printf("[For the initial step] What is the maximum number of iterations a solver can take? (q to exit)?\n");
	int itersToDo_init = getInt(1000, 1);
	// printf("[For the initial step] What should the MPC time budget be (in ms)? (q to exit)?\n");
	int timeLimit_init = 1000; //getInt(1000, 1); //note in ms
	printf("[For the figure eight] What is the maximum number of iterations a solver can take? (q to exit)?\n");
	int itersToDo = getInt(1000, 1);
	// printf("[For the figure eight] What should the MPC time budget be (in ms)? (q to exit)?\n");
	int timeLimit = 10000; //getInt(1000, 1); //note in ms
	// get the total traj time
	printf("How many seconds long should one figure eight of the tracked trajectory be? (q to exit)\n");
	double totalTime_us = 1000000.0*static_cast<double>(getInt(100, 1));
	// allocate variables and load inital variables
	trajVars<T> *tvars = new trajVars<T>; matDimms *dimms = new matDimms; algTrace<T> *atrace = new algTrace<T>;
	costParams<T> *cst = new costParams<T>;	loadCost(cst); // load in default cost to start
    std::thread mpcThread; LCM_MPCLoop_Handler<T> *mpchandler; CPUVars<T> *cvars; GPUVars<T> *gvars; // pointers for reference later
    // allocate for CPU / GPU
    if (mode == 'G'){gvars = new GPUVars<T>; allocateMemory_GPU_MPC<T>(gvars, dimms, tvars);}
    else{		     cvars = new CPUVars<T>; allocateMemory_CPU_MPC<T>(cvars, dimms, tvars);}
    // get the goal handler
    LCM_Fig8Goal_Handler<T> *goalhandler = new LCM_Fig8Goal_Handler<T>(totalTime_us, E_NORM_LIM, V_NORM_LIM, itersToDo, timeLimit);
    // then load the goals and LCM handlers and launch the MPC threads
    if (mode == 'G'){
    	// load initial traj and goal and run to full convergence to warm start
    	loadTraj<T>(gvars, tvars, dimms, xInit);	goalhandler->loadInitialGoal(gvars->xGoal);		goalhandler->loadInitialTarget(gvars->xTarget,xInit);
    	runiLQR_MPC_GPU<T>(tvars,gvars,dimms,atrace,cst,0,0,1);
		// then create the handler and launch the MPC thread
		mpchandler = new LCM_MPCLoop_Handler<T>(gvars,tvars,dimms,atrace,cst,itersToDo_init,timeLimit_init);
     	mpcThread  = std::thread(&runMPCHandler<T>, mpchandler);    
    }
    else{
    	// load initial goal and run to full convergence to warm start
    	loadTraj<T>(cvars, tvars, dimms, xInit);	goalhandler->loadInitialGoal(cvars->xGoal);		goalhandler->loadInitialTarget(cvars->xTarget,xInit);
    	runiLQR_MPC_CPU<T>(tvars,cvars,dimms,atrace,cst,0,0,1);
		// then create the handler and launch the MPC thread
		mpchandler = new LCM_MPCLoop_Handler<T>(cvars,tvars,dimms,atrace,cst,itersToDo_init,timeLimit_init);
     	mpcThread  = std::thread(&runMPCHandler<T>, mpchandler);   
     	if(FORCE_CORE_SWITCHES){setCPUForThread(&mpcThread, 1);} // move to another CPU
    }
    // launch the goal monitor
    std::thread goalThread = std::thread(&runFig8GoalLCM<T>, goalhandler);
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
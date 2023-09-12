/****************************************************************
 * CUDA Rigid Body DYNamics
 *
 * Based on the Joint Space Inversion Algorithm
 * currently special cased for 7dof Kuka Arm
 *
 * initI(T *s_I), initT(T *s_T)
 *
 * compute_eePos(T *s_T, T *s_eePos, T *s_dT, T *s_deePos, T *s_sinq, T *s_Tb, T *s_dTb, T *s_x, T *s_cosq, T *d_Tbody);
 *   (load_Tb, compute_T_TA_J, compute_dT_dTA_dJ)
 *
 * dynamics(T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody, T *s_eePos = nullptr, int reps = 1);
 *    load_Tb, load_I
 *    compute_T_TA_J
 *      loadAdjoint
 *    (compute_eePos)
 *    compute_Iw_Icrbs_twist
 *    compute_JdotV
 *      crfm
 *    compute_M_Tau
 *      crfm
 *    invertMatrix
 *    compute_qdd
 *
 * dynamicsGradient(T *s_dqdd, T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody);
 *    load_Tb, load_I
 *    compute_T_TA_J, compute_dT_dTA_dJ
 *      loadAdjoint
 *    (compute_eePos)
 *    compute_Iw_Icrbs_twist
 *    compute_JdotV
 *      crfm
 *    compute_M_Tau
 *      crfm
 *    invertMatrix, compute_qdd
 *    compute_dM, compute_dqdd_dM
 *    compute_dtwist, compute_dJdotV, compute_dWb
 *       crfm, crfmz
 *    compute_dTau, finish_dqdd
 *****************************************************************/

#if MPC_MODE
   #define GRAVITY 0 // Kuka arm does automatic gravity compensation on top of torques sent so assume 0
#else
   #define GRAVITY 9.81 // in full iLQR sim we are trying to come up with something assuming no gravity comp or anything hardware related
#endif
/*** PLANT SPECIFIC RBDYN HELPERS ***/
#define EE_ON_LINK_X 0
#define EE_ON_LINK_Y 0
#ifndef EE_TYPE
    #define EE_TYPE 1 // by default the arm has a flange and no end effector
#endif
#if EE_TYPE == 0 // no ee
    #define EE_ON_LINK_Z 0
    #define INERTIA_MODIFIER 1
    #define WEIGHT_MODIFIER 0
#elif EE_TYPE == 1 // flange only (2.5 inches)
    #define EE_ON_LINK_Z 0.0635
    #define INERTIA_MODIFIER 3
    #define WEIGHT_MODIFIER 0.03
#elif EE_TYPE == 2 // flange + peg (6 inches)
    #define EE_ON_LINK_Z 0.1524
    #define INERTIA_MODIFIER 5
    #define WEIGHT_MODIFIER 0.5
#endif

#if USE_EE_VEL_COST
    #warning "EE_VEL for rpy is broken at this time please do not use in your algorithm\n"
#endif

template <typename T>
__host__ __device__ 
void initI(T *s_I){
    #pragma unroll
    for (int i = 0; i < 36*NUM_POS; i++){s_I[i] = static_cast<T>(0);}
    #if USE_WAFR_URDF
        s_I[0] = static_cast<T>(0.121128);
        s_I[4] = static_cast<T>(-0.6912);
        s_I[5] = static_cast<T>(-0.1728);
        s_I[7] = static_cast<T>(0.116244);
        s_I[8] = static_cast<T>(0.025623);
        s_I[9] = static_cast<T>(0.6912);
        s_I[13] = static_cast<T>(0.025623);
        s_I[14] = static_cast<T>(0.017484);
        s_I[15] = static_cast<T>(0.1728);
        s_I[19] = static_cast<T>(0.6912);
        s_I[20] = static_cast<T>(0.1728);
        s_I[21] = static_cast<T>(5.76);
        s_I[24] = static_cast<T>(-0.6912);
        s_I[28] = static_cast<T>(5.76);
        s_I[30] = static_cast<T>(-0.1728);
        s_I[35] = static_cast<T>(5.76);
        s_I[36] = static_cast<T>(0.06380575);
        s_I[37] = static_cast<T>(-0.000112395);
        s_I[38] = static_cast<T>(-0.00008001);
        s_I[40] = static_cast<T>(-0.2667);
        s_I[41] = static_cast<T>(0.37465);
        s_I[42] = static_cast<T>(-0.000112395);
        s_I[43] = static_cast<T>(0.0416019715);
        s_I[44] = static_cast<T>(-0.0108483);
        s_I[45] = static_cast<T>(0.2667);
        s_I[47] = static_cast<T>(-0.001905);
        s_I[48] = static_cast<T>(-0.00008001);
        s_I[49] = static_cast<T>(-0.0108483);
        s_I[50] = static_cast<T>(0.0331049215);
        s_I[51] = static_cast<T>(-0.37465);
        s_I[52] = static_cast<T>(0.001905);
        s_I[55] = static_cast<T>(0.2667);
        s_I[56] = static_cast<T>(-0.37465);
        s_I[57] = static_cast<T>(6.35);
        s_I[60] = static_cast<T>(-0.2667);
        s_I[62] = static_cast<T>(0.001905);
        s_I[66] = static_cast<T>(0.37465);
        s_I[67] = static_cast<T>(-0.001905);
        s_I[71] = static_cast<T>(6.35);
        s_I[72] = static_cast<T>(0.0873);
        s_I[76] = static_cast<T>(-0.455);
        s_I[77] = static_cast<T>(0.105);
        s_I[79] = static_cast<T>(0.08295);
        s_I[80] = static_cast<T>(-0.00878);
        s_I[81] = static_cast<T>(0.455);
        s_I[85] = static_cast<T>(-0.00878);
        s_I[86] = static_cast<T>(0.01075);
        s_I[87] = static_cast<T>(-0.105);
        s_I[64] = static_cast<T>(6.35);
        s_I[91] = static_cast<T>(0.455);
        s_I[92] = static_cast<T>(-0.105);
        s_I[93] = static_cast<T>(3.5);
        s_I[96] = static_cast<T>(-0.455);
        s_I[100] = static_cast<T>(3.5);
        s_I[102] = static_cast<T>(0.105);
        s_I[107] = static_cast<T>(3.5);
        s_I[108] = static_cast<T>(0.0367575);
        s_I[112] = static_cast<T>(-0.119);
        s_I[113] = static_cast<T>(0.2345);
        s_I[115] = static_cast<T>(0.020446);
        s_I[116] = static_cast<T>(-0.005133);
        s_I[117] = static_cast<T>(0.119);
        s_I[121] = static_cast<T>(-0.005133);
        s_I[122] = static_cast<T>(0.0217115);
        s_I[123] = static_cast<T>(-0.2345);
        s_I[127] = static_cast<T>(0.119);
        s_I[128] = static_cast<T>(-0.2345);
        s_I[129] = static_cast<T>(3.5);
        s_I[132] = static_cast<T>(-0.119);
        s_I[136] = static_cast<T>(3.5);
        s_I[138] = static_cast<T>(0.2345);
        s_I[143] = static_cast<T>(3.5);
        s_I[144] = static_cast<T>(0.0317595);
        s_I[145] = static_cast<T>(-0.00000735);
        s_I[146] = static_cast<T>(-0.0000266);
        s_I[148] = static_cast<T>(-0.266);
        s_I[149] = static_cast<T>(0.0735);
        s_I[150] = static_cast<T>(-0.00000735);
        s_I[151] = static_cast<T>(0.028916035);
        s_I[152] = static_cast<T>(-0.002496);
        s_I[153] = static_cast<T>(0.266);
        s_I[155] = static_cast<T>(-0.00035);
        s_I[156] = static_cast<T>(-0.0000266);
        s_I[157] = static_cast<T>(-0.002496);
        s_I[158] = static_cast<T>(0.006033535);
        s_I[159] = static_cast<T>(-0.0735);
        s_I[160] = static_cast<T>(0.00035);
        s_I[163] = static_cast<T>(0.266);
        s_I[164] = static_cast<T>(-0.0735);
        s_I[165] = static_cast<T>(3.5);
        s_I[168] = static_cast<T>(-0.266);
        s_I[170] = static_cast<T>(0.00035);
        s_I[172] = static_cast<T>(3.5);
        s_I[174] = static_cast<T>(0.0735);
        s_I[175] = static_cast<T>(-0.00035);
        s_I[179] = static_cast<T>(3.5);
        s_I[180] = static_cast<T>(0.004900936);
        s_I[184] = static_cast<T>(-0.00072);
        s_I[185] = static_cast<T>(0.00108);
        s_I[187] = static_cast<T>(0.004700288);
        s_I[188] = static_cast<T>(0.000245568);
        s_I[189] = static_cast<T>(0.00072);
        s_I[193] = static_cast<T>(0.000245568);
        s_I[194] = static_cast<T>(0.003600648);
        s_I[195] = static_cast<T>(-0.00108);
        s_I[199] = static_cast<T>(0.00072);
        s_I[200] = static_cast<T>(-0.00108);
        s_I[201] = static_cast<T>(1.8);
        s_I[204] = static_cast<T>(-0.00072);
        s_I[208] = static_cast<T>(1.8);
        s_I[210] = static_cast<T>(0.00108);
        s_I[215] = static_cast<T>(1.8);
        s_I[216] = static_cast<T>(0.10732607081630547718464896433943);
        s_I[217] = static_cast<T>(0.00019989199349797965904636243283932);
        s_I[218] = static_cast<T>(-0.0053829202963780343332844680048765);
        s_I[219] = static_cast<T>(-0.000000000000000000000000086572214906444569626737234912971);
        s_I[220] = static_cast<T>(-0.4471958408698656350921396551712);
        s_I[221] = static_cast<T>(-0.00000026790984812577454527075348882093);
        s_I[222] = static_cast<T>(0.00019989199349797960483625380856409);
        s_I[223] = static_cast<T>(0.098340754312488648514190003879776);
        s_I[224] = static_cast<T>(0.000029124896308221664471436659904491);
        s_I[225] = static_cast<T>(0.4471958408698656350921396551712);
        s_I[226] = static_cast<T>(-0.000000000000000000000037314314248140254118040921485142);
        s_I[227] = static_cast<T>(0.012494963104108957122062584232935);
        s_I[228] = static_cast<T>(-0.005382920296378033465922730016473);
        s_I[229] = static_cast<T>(0.000029124896308221881311871157005378);
        s_I[230] = static_cast<T>(0.12444928521768228169008807526552);
        s_I[231] = static_cast<T>(0.00000026790984812750932168628949930911);
        s_I[232] = static_cast<T>(-0.012494963104108946713721728372093);
        s_I[233] = static_cast<T>(-0.0000000000000000000033881317890178754317538255352671);
        s_I[234] = static_cast<T>(-0.0000000000000000000016941561609528859931138471397541);
        s_I[235] = static_cast<T>(0.4471958408698656350921396551712);
        s_I[236] = static_cast<T>(0.00000026790984812750932168628949930911);
        s_I[237] = static_cast<T>(6.4);
        s_I[238] = static_cast<T>(0.00000000000000000000000091178226550731791047478487008133);
        s_I[239] = static_cast<T>(0.00000000000000000010842021729662189555389373538375);
        s_I[240] = static_cast<T>(-0.4471958408698656350921396551712);
        s_I[241] = static_cast<T>(-0.000000000000000000000051018957034122961580356685370858);
        s_I[242] = static_cast<T>(-0.0124949631041089484484452043489);
        s_I[243] = static_cast<T>(0.00000000000000000000000087446681723681373628459234511013);
        s_I[244] = static_cast<T>(6.4);
        s_I[245] = static_cast<T>(-0.00000000000000000000084498302793689761931173291132555);
        s_I[246] = static_cast<T>(-0.00000026790984812924404516226630640352);
        s_I[247] = static_cast<T>(0.012494963104108955387339108256128);
        s_I[248] = static_cast<T>(-0.0000000000000000000000000000000004);
        s_I[249] = static_cast<T>(0.00000000000000000011);
        s_I[250] = static_cast<T>(-0.0000000000000000000007);
        s_I[251] = static_cast<T>(6.4);
    #else
        s_I[0] = static_cast<T>(0.12112799999999998568078751759458);
        s_I[4] = static_cast<T>(-0.6912);
        s_I[5] = static_cast<T>(-0.1728);
        s_I[7] = static_cast<T>(0.11624399999999998622790542412986);
        s_I[8] = static_cast<T>(0.020736);
        s_I[9] = static_cast<T>(0.6912);
        s_I[13] = static_cast<T>(0.020736);
        s_I[14] = static_cast<T>(0.017483999999999999541699935434735);
        s_I[15] = static_cast<T>(0.1728);
        s_I[19] = static_cast<T>(0.6912);
        s_I[20] = static_cast<T>(0.1728);
        s_I[21] = static_cast<T>(5.76);
        s_I[24] = static_cast<T>(-0.6912);
        s_I[28] = static_cast<T>(5.76);
        s_I[30] = static_cast<T>(-0.1728);
        s_I[35] = static_cast<T>(5.76);
        s_I[36] = static_cast<T>(0.063805749999999994415134096925613);
        s_I[37] = static_cast<T>(-0.00011239499999999998352661484402049);
        s_I[38] = static_cast<T>(-0.000080010000000000001366302904148853);
        s_I[40] = static_cast<T>(-0.2667);
        s_I[41] = static_cast<T>(0.37465);
        s_I[42] = static_cast<T>(-0.00011239499999999998352661484402049);
        s_I[43] = static_cast<T>(0.041601971500000001213948053191416);
        s_I[44] = static_cast<T>(-0.015735300000000000675282052498005);
        s_I[45] = static_cast<T>(0.2667);
        s_I[47] = static_cast<T>(-0.0019049999999999997924576833341348);
        s_I[48] = static_cast<T>(-0.000080009999999999987813775748080047);
        s_I[49] = static_cast<T>(-0.015735299999999997205835100544391);
        s_I[50] = static_cast<T>(0.033104921499999995226914961676812);
        s_I[51] = static_cast<T>(-0.37465);
        s_I[52] = static_cast<T>(0.0019049999999999997924576833341348);
        s_I[55] = static_cast<T>(0.2667);
        s_I[56] = static_cast<T>(-0.37465);
        s_I[57] = static_cast<T>(6.35);
        s_I[60] = static_cast<T>(-0.2667);
        s_I[62] = static_cast<T>(0.0019049999999999997924576833341348);
        s_I[64] = static_cast<T>(6.35);
        s_I[66] = static_cast<T>(0.37465);
        s_I[67] = static_cast<T>(-0.0019049999999999997924576833341348);
        s_I[71] = static_cast<T>(6.35);
        s_I[72] = static_cast<T>(0.0873);
        s_I[76] = static_cast<T>(-0.455);
        s_I[77] = static_cast<T>(0.105);
        s_I[79] = static_cast<T>(0.08295);
        s_I[80] = static_cast<T>(-0.01365);
        s_I[81] = static_cast<T>(0.455);
        s_I[85] = static_cast<T>(-0.01365);
        s_I[86] = static_cast<T>(0.01075);
        s_I[87] = static_cast<T>(-0.105);
        s_I[91] = static_cast<T>(0.455);
        s_I[92] = static_cast<T>(-0.105);
        s_I[93] = static_cast<T>(3.5);
        s_I[96] = static_cast<T>(-0.455);
        s_I[100] = static_cast<T>(3.5);
        s_I[102] = static_cast<T>(0.105);
        s_I[107] = static_cast<T>(3.5);
        s_I[108] = static_cast<T>(0.036757500000000005446754158811018);
        s_I[112] = static_cast<T>(-0.119);
        s_I[113] = static_cast<T>(0.2345);
        s_I[115] = static_cast<T>(0.020446000000000002366773443895909);
        s_I[116] = static_cast<T>(-0.0079730000000000009197087535994797);
        s_I[117] = static_cast<T>(0.119);
        s_I[121] = static_cast<T>(-0.0079730000000000009197087535994797);
        s_I[122] = static_cast<T>(0.02171150000000000163113966777928);
        s_I[123] = static_cast<T>(-0.2345);
        s_I[127] = static_cast<T>(0.119);
        s_I[128] = static_cast<T>(-0.2345);
        s_I[129] = static_cast<T>(3.5);
        s_I[132] = static_cast<T>(-0.119);
        s_I[136] = static_cast<T>(3.5);
        s_I[138] = static_cast<T>(0.2345);
        s_I[143] = static_cast<T>(3.5);
        s_I[144] = static_cast<T>(0.031759500000000003006039861475074);
        s_I[145] = static_cast<T>(-0.0000073500000000000007791293651915332);
        s_I[146] = static_cast<T>(-0.0000265999999999999994315744850093);
        s_I[148] = static_cast<T>(-0.266);
        s_I[149] = static_cast<T>(0.0735);
        s_I[150] = static_cast<T>(-0.0000073500000000000016261623124458335);
        s_I[151] = static_cast<T>(0.028916034999999999655084792493653);
        s_I[152] = static_cast<T>(-0.0055860000000000006870060076380469);
        s_I[153] = static_cast<T>(0.266);
        s_I[155] = static_cast<T>(-0.00035);
        s_I[156] = static_cast<T>(-0.000026600000000000002819706274026501);
        s_I[157] = static_cast<T>(-0.0055860000000000006870060076380469);
        s_I[158] = static_cast<T>(0.0060335350000000004602740411030481);
        s_I[159] = static_cast<T>(-0.0735);
        s_I[160] = static_cast<T>(0.00035);
        s_I[163] = static_cast<T>(0.266);
        s_I[164] = static_cast<T>(-0.0735);
        s_I[165] = static_cast<T>(3.5);
        s_I[168] = static_cast<T>(-0.266);
        s_I[170] = static_cast<T>(0.00035);
        s_I[172] = static_cast<T>(3.5);
        s_I[174] = static_cast<T>(0.0735);
        s_I[175] = static_cast<T>(-0.00035);
        s_I[179] = static_cast<T>(3.5);
        s_I[180] = static_cast<T>(0.0049009359999999998341868590046033);
        s_I[184] = static_cast<T>(-0.00072);
        s_I[185] = static_cast<T>(0.00108);
        s_I[187] = static_cast<T>(0.0047002880000000003823945604608525);
        s_I[188] = static_cast<T>(-0.00000043200000000000000374459035827612);
        s_I[189] = static_cast<T>(0.00072);
        s_I[193] = static_cast<T>(-0.00000043200000000000000374459035827612);
        s_I[194] = static_cast<T>(0.0036006479999999996960413639612852);
        s_I[195] = static_cast<T>(-0.00108);
        s_I[199] = static_cast<T>(0.00072);
        s_I[200] = static_cast<T>(-0.00108);
        s_I[201] = static_cast<T>(1.8);
        s_I[204] = static_cast<T>(-0.00072);
        s_I[208] = static_cast<T>(1.8);
        s_I[210] = static_cast<T>(0.00108);
        s_I[215] = static_cast<T>(1.8);
        s_I[216] = static_cast<T>(0.0055*INERTIA_MODIFIER);
        s_I[220] = static_cast<T>(-0.024*INERTIA_MODIFIER);
        s_I[223] = static_cast<T>(0.0055*INERTIA_MODIFIER);
        s_I[225] = static_cast<T>(0.024*INERTIA_MODIFIER);
        s_I[230] = static_cast<T>(0.005*INERTIA_MODIFIER);
        s_I[235] = static_cast<T>(0.024*INERTIA_MODIFIER);
        s_I[237] = static_cast<T>(1.2+WEIGHT_MODIFIER);
        s_I[240] = static_cast<T>(-0.024*INERTIA_MODIFIER);
        s_I[244] = static_cast<T>(1.2+WEIGHT_MODIFIER);
        s_I[251] = static_cast<T>(1.2+WEIGHT_MODIFIER);
    #endif
}

template <typename T>
__host__ __device__ 
void initT(T *s_T){
    #pragma unroll
    for (int i = 0; i < 36*NUM_POS; i++){s_T[i] = 0.0;}
    #if USE_WAFR_URDF
        s_T[10] = static_cast<T>(1.0);
        s_T[14] = static_cast<T>(0.1575);
        s_T[15] = static_cast<T>(1.0);

        s_T[44] = static_cast<T>(-0.00000000000020682);
        s_T[45] = static_cast<T>(1.0);
        s_T[46] = static_cast<T>(0.0000000000048966);
        s_T[50] = static_cast<T>(0.2025);
        s_T[51] = static_cast<T>(1.0);

        s_T[80] = static_cast<T>(-0.00000000000020682);
        s_T[81] = static_cast<T>(1.0);
        s_T[82] = static_cast<T>(0.0000000000048966);
        s_T[85] = static_cast<T>(0.2045);
        s_T[87] = static_cast<T>(1.0);

        s_T[117] = static_cast<T>(-1.0);
        s_T[118] = static_cast<T>(0.0000000000048966);
        s_T[122] = static_cast<T>(0.2155);
        s_T[123] = static_cast<T>(1.0);

        s_T[152] = static_cast<T>(-0.0000000000000000000000010127);
        s_T[153] = static_cast<T>(1.0);
        s_T[154] = static_cast<T>(-0.0000000000048966);
        s_T[157] = static_cast<T>(0.1845);
        s_T[159] = static_cast<T>(1.0);

        s_T[189] = static_cast<T>(-1.0);
        s_T[190] = static_cast<T>(0.0000000000048966);
        s_T[194] = static_cast<T>(0.2155);
        s_T[195] = static_cast<T>(1.0);

        s_T[224] = static_cast<T>(-0.0000000000000000000000010127);
        s_T[225] = static_cast<T>(1.0);
        s_T[226] = static_cast<T>(-0.0000000000048966);
        s_T[229] = static_cast<T>(0.081);
        s_T[231] = static_cast<T>(1.0);
    #else
        s_T[10] = static_cast<T>(1.0);
        s_T[14] = static_cast<T>(0.1575);
        s_T[15] = static_cast<T>(1.0);
        s_T[44] = static_cast<T>(0.00000000000000012246467991473532071737640294584);
        s_T[45] = static_cast<T>(1.0);
        s_T[46] = static_cast<T>(-0.0000000000000003828568698926949434848128216851);
        s_T[50] = static_cast<T>(0.2025);
        s_T[51] = static_cast<T>(1.0);
        s_T[80] = static_cast<T>(0.00000000000000012246467991473532071737640294584);
        s_T[81] = static_cast<T>(1.0);
        s_T[82] = static_cast<T>(-0.0000000000000003828568698926949434848128216851);
        s_T[85] = static_cast<T>(0.2045);
        s_T[87] = static_cast<T>(1.0);
        s_T[117] = static_cast<T>(-1.0);
        s_T[118] = static_cast<T>(-0.0000000000000003828568698926949434848128216851);
        s_T[122] = static_cast<T>(0.2155);
        s_T[123] = static_cast<T>(1.0);
        s_T[152] = static_cast<T>(-0.000000000000000000000000000000046886444024566354179237414162174);
        s_T[153] = static_cast<T>(1.0);
        s_T[154] = static_cast<T>(0.0000000000000003828568698926949434848128216851);
        s_T[157] = static_cast<T>(0.1845);
        s_T[159] = static_cast<T>(1.0);
        s_T[189] = static_cast<T>(-1.0);
        s_T[190] = static_cast<T>(-0.0000000000000003828568698926949434848128216851);
        s_T[194] = static_cast<T>(0.2155);
        s_T[195] = static_cast<T>(1.0);
        s_T[224] = static_cast<T>(-0.000000000000000000000000000000046886444024566354179237414162174);
        s_T[225] = static_cast<T>(1.0);
        s_T[226] = static_cast<T>(0.0000000000000003828568698926949434848128216851);
        s_T[229] = static_cast<T>(0.081);
        s_T[231] = static_cast<T>(1.0);
    #endif
}

template <typename T>
__host__ __device__ 
void updateT(T *s_T, T *s_cosx, T *s_sinx, int ld = 36){
    #if USE_WAFR_URDF
        s_T[0] = s_cosx[0];
        s_T[1] = s_sinx[0];
        s_T[4] = -s_sinx[0];
        s_T[5] = s_cosx[0];

        s_T[ld]   = static_cast<T>(0.0000000000000000000000010127)*s_sinx[1]-s_cosx[1];
        s_T[ld+1] = static_cast<T>(-0.00000000000020682)*s_cosx[1]-static_cast<T>(0.0000000000048966)*s_sinx[1];
        s_T[ld+2] = s_sinx[1];
        s_T[ld+4] = static_cast<T>(0.0000000000000000000000010127)*s_cosx[1]+s_sinx[1];
        s_T[ld+5] = static_cast<T>(0.00000000000020682)*s_sinx[1]-static_cast<T>(0.0000000000048966)*s_cosx[1];
        s_T[ld+6] = s_cosx[1];
        s_T[ld+8] = static_cast<T>(-0.00000000000020682);

        s_T[2*ld]   = static_cast<T>(0.0000000000000000000000010127)*s_sinx[2]-s_cosx[2];
        s_T[2*ld+1] = static_cast<T>(-0.00000000000020682)*s_cosx[2]-static_cast<T>(0.0000000000048966)*s_sinx[2];
        s_T[2*ld+2] = s_sinx[2];
        s_T[2*ld+4] = static_cast<T>(0.0000000000000000000000010127)*s_cosx[2]+s_sinx[2];
        s_T[2*ld+5] = static_cast<T>(0.00000000000020682)*s_sinx[2]-static_cast<T>(0.0000000000048966)*s_cosx[2];
        s_T[2*ld+6] = s_cosx[2];

        s_T[3*ld]   = s_cosx[3];
        s_T[3*ld+1] = static_cast<T>(0.0000000000048966)*s_sinx[3];
        s_T[3*ld+2] = s_sinx[3];
        s_T[3*ld+4] = -s_sinx[3];
        s_T[3*ld+5] = static_cast<T>(0.0000000000048966)*s_cosx[3];
        s_T[3*ld+6] = s_cosx[3];

        s_T[4*ld]   = static_cast<T>(0.00000000000020682)*s_sinx[4]-s_cosx[4];
        s_T[4*ld+1] = static_cast<T>(0.0000000000048966)*s_sinx[4];
        s_T[4*ld+2] = static_cast<T>(0.00000000000020682)*s_cosx[4]+s_sinx[4];
        s_T[4*ld+4] = static_cast<T>(0.00000000000020682)*s_cosx[4]+s_sinx[4];
        s_T[4*ld+5] = static_cast<T>(0.0000000000048966)*s_cosx[4];
        s_T[4*ld+6] = static_cast<T>(-0.00000000000020682)*s_sinx[4]+s_cosx[4];

        s_T[5*ld]   = s_cosx[5];
        s_T[5*ld+1] = static_cast<T>(0.0000000000048966)*s_sinx[5];
        s_T[5*ld+2] = s_sinx[5];
        s_T[5*ld+4] = -s_sinx[5];
        s_T[5*ld+5] = static_cast<T>(0.0000000000048966)*s_cosx[5];
        s_T[5*ld+6] = s_cosx[5];

        s_T[6*ld]   = static_cast<T>(0.00000000000020682)*s_sinx[6]-s_cosx[6];
        s_T[6*ld+1] = static_cast<T>(0.0000000000048966)*s_sinx[6];
        s_T[6*ld+2] = static_cast<T>(0.00000000000020682)*s_cosx[6]+s_sinx[6];
        s_T[6*ld+4] = static_cast<T>(0.00000000000020682)*s_cosx[6]+s_sinx[6];
        s_T[6*ld+5] = static_cast<T>(0.0000000000048966)*s_cosx[6];
        s_T[6*ld+6] = static_cast<T>(-0.00000000000020682)*s_sinx[6]+s_cosx[6];
    #else
        s_T[0] = s_cosx[0];
        s_T[1] = s_sinx[0];
        s_T[4] = -s_sinx[0];
        s_T[5] = s_cosx[0];
        s_T[ld] = static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_sinx[1] - s_cosx[1];
        s_T[ld+1] = static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[1] + static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[1];
        s_T[ld+2] = s_sinx[1];
        s_T[ld+4] = static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_cosx[1] + s_sinx[1];
        s_T[ld+5] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_cosx[1] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[1];
        s_T[ld+6] = s_cosx[1];
        s_T[2*ld]   = static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_sinx[2] - 1.0*s_cosx[2];
        s_T[2*ld+1] = static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[2] + static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[2];
        s_T[2*ld+2] = s_sinx[2];
        s_T[2*ld+4] = static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_cosx[2] + s_sinx[2];
        s_T[2*ld+5] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_cosx[2] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[2];
        s_T[2*ld+6] = s_cosx[2];
        s_T[3*ld]   = s_cosx[3];
        s_T[3*ld+1] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_sinx[3];
        s_T[3*ld+2] = s_sinx[3];
        s_T[3*ld+4] = -s_sinx[3];
        s_T[3*ld+5] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[3];
        s_T[3*ld+6] = s_cosx[3];
        s_T[4*ld]   = -s_cosx[4] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[4];
        s_T[4*ld+1] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_sinx[4];
        s_T[4*ld+2] = s_sinx[4] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[4];
        s_T[4*ld+4] = s_sinx[4] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[4];
        s_T[4*ld+5] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[4];
        s_T[4*ld+6] = s_cosx[4] + static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[4];
        s_T[5*ld]   = s_cosx[5];
        s_T[5*ld+1] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_sinx[5];
        s_T[5*ld+2] = s_sinx[5];
        s_T[5*ld+4] = -s_sinx[5];
        s_T[5*ld+5] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[5];
        s_T[5*ld+6] = s_cosx[5];
        s_T[6*ld]   = -s_cosx[6] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[6];
        s_T[6*ld+1] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_sinx[6];
        s_T[6*ld+2] = s_sinx[6] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[6];
        s_T[6*ld+4] = s_sinx[6] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[6];
        s_T[6*ld+5] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[6];
        s_T[6*ld+6] = s_cosx[6] + static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[6];
    #endif
}

template <typename T>
__host__ __device__ 
void loadTdx4(T *s_Tdx, T *s_cosx, T *s_sinx){
    #pragma unroll
    for (int i = 0; i < 16*NUM_POS; i++){s_Tdx[i] = 0.0;}
    #if USE_WAFR_URDF
        s_Tdx[0] = -s_sinx[0];
        s_Tdx[1] = s_cosx[0];
        s_Tdx[4] = -s_cosx[0];
        s_Tdx[5] = -s_sinx[0];
        s_Tdx[16] = static_cast<T>(0.0000000000000000000000010127)*s_cosx[1]+s_sinx[1];
        s_Tdx[17] = static_cast<T>(0.00000000000020682)*s_sinx[1]-static_cast<T>(0.0000000000048966)*s_cosx[1];
        s_Tdx[18] = s_cosx[1];
        s_Tdx[20] = static_cast<T>(-0.0000000000000000000000010127)*s_sinx[1]+s_cosx[1];
        s_Tdx[21] = static_cast<T>(0.00000000000020682)*s_cosx[1]+static_cast<T>(0.0000000000048966)*s_sinx[1];
        s_Tdx[22] = -s_sinx[1];
        s_Tdx[32] = static_cast<T>(0.0000000000000000000000010127)*s_cosx[2]+s_sinx[2];
        s_Tdx[33] = static_cast<T>(0.00000000000020682)*s_sinx[2]-static_cast<T>(0.0000000000048966)*s_cosx[2];
        s_Tdx[34] = s_cosx[2];
        s_Tdx[36] = static_cast<T>(-0.0000000000000000000000010127)*s_sinx[2]+s_cosx[2];
        s_Tdx[37] = static_cast<T>(0.00000000000020682)*s_cosx[2]+static_cast<T>(0.0000000000048966)*s_sinx[2];
        s_Tdx[38] = -s_sinx[2];
        s_Tdx[48] = -s_sinx[3];
        s_Tdx[49] = static_cast<T>(0.0000000000048966)*s_cosx[3];
        s_Tdx[50] = s_cosx[3];
        s_Tdx[52] = -s_cosx[3];
        s_Tdx[53] = static_cast<T>(-0.0000000000048966)*s_sinx[3];
        s_Tdx[54] = -s_sinx[3];
        s_Tdx[64] = static_cast<T>(0.00000000000020682)*s_cosx[4]+s_sinx[4];
        s_Tdx[65] = static_cast<T>(0.0000000000048966)*s_cosx[4];
        s_Tdx[66] = static_cast<T>(-0.00000000000020682)*s_sinx[4]+s_cosx[4];
        s_Tdx[68] = static_cast<T>(-0.00000000000020682)*s_sinx[4]+s_cosx[4];
        s_Tdx[69] = static_cast<T>(-0.0000000000048966)*s_sinx[4];
        s_Tdx[70] = static_cast<T>(-0.00000000000020682)*s_cosx[4]-s_sinx[4];
        s_Tdx[80] = -s_sinx[5];
        s_Tdx[81] = static_cast<T>(0.0000000000048966)*s_cosx[5];
        s_Tdx[82] = s_cosx[5];
        s_Tdx[84] = -s_cosx[5];
        s_Tdx[85] = static_cast<T>(-0.0000000000048966)*s_sinx[5];
        s_Tdx[86] = -s_sinx[5];
        s_Tdx[96] = static_cast<T>(0.00000000000020682)*s_cosx[6]+s_sinx[6];
        s_Tdx[97] = static_cast<T>(0.0000000000048966)*s_cosx[6];
        s_Tdx[98] = static_cast<T>(-0.00000000000020682)*s_sinx[6]+s_cosx[6];
        s_Tdx[100] = static_cast<T>(-0.00000000000020682)*s_sinx[6]+s_cosx[6];
        s_Tdx[101] = static_cast<T>(-0.0000000000048966)*s_sinx[6];
        s_Tdx[102] = static_cast<T>(-0.00000000000020682)*s_cosx[6]-s_sinx[6];
    #else
        s_Tdx[0] = -s_sinx[0];
        s_Tdx[1] = s_cosx[0];
        s_Tdx[4] = -s_cosx[0];
        s_Tdx[5] = -s_sinx[0];
        s_Tdx[16] = static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_cosx[1] + s_sinx[1];
        s_Tdx[17] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_cosx[1] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[1];
        s_Tdx[18] = s_cosx[1];
        s_Tdx[20] = s_cosx[1] - static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_sinx[1];
        s_Tdx[21] = static_cast<T>(-0.00000000000000012246467991473532071737640294584)*s_cosx[1] - static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[1];
        s_Tdx[22] = -s_sinx[1];
        s_Tdx[32] = static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_cosx[2] + s_sinx[2];
        s_Tdx[33] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_cosx[2] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[2];
        s_Tdx[34] = s_cosx[2];
        s_Tdx[36] = s_cosx[2] - static_cast<T>(0.000000000000000000000000000000046886444024566354179237414162174)*s_sinx[2];
        s_Tdx[37] = static_cast<T>(-0.00000000000000012246467991473532071737640294584)*s_cosx[2] - static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[2];
        s_Tdx[38] = -s_sinx[2];
        s_Tdx[48] = -s_sinx[3];
        s_Tdx[49] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[3];
        s_Tdx[50] = s_cosx[3];
        s_Tdx[52] = -s_cosx[3];
        s_Tdx[53] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[3];
        s_Tdx[54] = -s_sinx[3];
        s_Tdx[64] = s_sinx[4] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[4];
        s_Tdx[65] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[4];
        s_Tdx[66] = s_cosx[4] + static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[4];
        s_Tdx[68] = s_cosx[4] + static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[4];
        s_Tdx[69] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[4];
        s_Tdx[70] = static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[4] - s_sinx[4];
        s_Tdx[80] = -s_sinx[5];
        s_Tdx[81] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[5];
        s_Tdx[82] = s_cosx[5];
        s_Tdx[84] = -s_cosx[5];
        s_Tdx[85] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[5];
        s_Tdx[86] = -s_sinx[5];
        s_Tdx[96] = s_sinx[6] - static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[6];
        s_Tdx[97] = static_cast<T>(-0.0000000000000003828568698926949434848128216851)*s_cosx[6];
        s_Tdx[98] = s_cosx[6] + static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[6];
        s_Tdx[100] = s_cosx[6] + static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_sinx[6];
        s_Tdx[101] = static_cast<T>(0.0000000000000003828568698926949434848128216851)*s_sinx[6];
        s_Tdx[102] = static_cast<T>(0.00000000000000012246467991473532071737640294584)*s_cosx[6] - s_sinx[6];
    #endif
}
/*** PLANT SPECIFIC RBDYN HELPERS ***/

/*** GENERAL PURPOSE RBDYN HELPERS ***/
template <typename T>
__host__ __device__ 
void loadAdjoint(T *dst, T *src){
   dst[0] = 0;
   dst[1] = src[2];
   dst[2] = -src[1];
   dst[3] = -src[2];
   dst[4] = 0;
   dst[5] = src[0];
   dst[6] = src[1];
   dst[7] = -src[0];
   dst[8] = 0;
}
template <typename T>
__host__ __device__ 
void loadAdjoint(T *dst, T src0, T src1, T src2){
   dst[0] = 0;
   dst[1] = src2;
   dst[2] = -src1;
   dst[3] = -src2;
   dst[4] = 0;
   dst[5] = src0;
   dst[6] = src1;
   dst[7] = -src0;
   dst[8] = 0;
}

template <typename T>
__host__ __device__ 
void crfm(T *dst, T *src, int f_flag){
   // don't think there is a better way to do this.... note make sure to clear first
   dst[1] = src[2];
   dst[2] = -src[1];
   dst[6] = -src[2];
   dst[8] = src[0];
   dst[12] = src[1];
   dst[13] = -src[0];
   dst[22] = src[2];
   dst[23] = -src[1];
   dst[27] = -src[2];
   dst[29] = src[0];
   dst[33] = src[1];
   dst[34] = -src[0];
   if (f_flag){
      dst[19] = src[5];
      dst[20] = -src[4];
      dst[24] = -src[5];
      dst[26] = src[3];
      dst[30] = src[4];
      dst[31] = -src[3];
   }
   else {
      dst[4] = src[5];
      dst[5] = -src[4];
      dst[9] = -src[5];
      dst[11] = src[3];
      dst[15] = src[4];
      dst[16] = -src[3];
   }
}

template <typename T>
__host__ __device__ 
void crfmz(T *dst, T *src, int f_flag){
   // this one is slower as it does all of the zeroing
   crfm(dst, src, f_flag);
   dst[0] = 0;
   dst[3] = 0;
   dst[7] = 0;
   dst[10] = 0;
   dst[14] = 0;
   dst[17] = 0;
   dst[18] = 0;
   dst[21] = 0;
   dst[25] = 0;
   dst[28] = 0;
   dst[32] = 0;
   dst[35] = 0;
   if (!f_flag){
      dst[19] = 0;
      dst[20] = 0;
      dst[24] = 0;
      dst[26] = 0;
      dst[30] = 0;
      dst[31] = 0;
   }
   else {
      dst[4] = 0;
      dst[5] = 0;
      dst[9] = 0;
      dst[11] = 0;
      dst[15] = 0;
      dst[16] = 0;
   }
}
/*** GENERAL PURPOSE RBDYN HELPERS ***/

/*** KINEMATICS AND DYNAMICS HELPERS ***/
template <typename T>
__host__ __device__ __forceinline__
void load_I(T *s_I, T *d_I){
   int start, delta; singleLoopVals(&start,&delta);
   for (int ind = start; ind < 36*NUM_POS; ind += delta){
      s_I[ind] = d_I[ind];
   }
}

template <typename T>
__host__ __device__ __forceinline__
void load_Tb(T *s_x, T *s_Tbody, T *d_Tbody, T *s_sinx, T *s_cosx, T *s_dTbody = nullptr){
   int start, delta; singleLoopVals(&start,&delta);
   // compute sin/cos in parallel as well as Tbase
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
         s_sinx[ind] = sin(s_x[ind]);    
         s_cosx[ind] = cos(s_x[ind]);
   }
   #pragma unroll
   for (int ind = start; ind < 36*NUM_POS; ind += delta){
      // #ifdef __CUDA_ARCH__
         s_Tbody[ind] = d_Tbody[ind];
      // #endif
      if (s_dTbody != nullptr){s_dTbody[ind] = 0;} // if need to load dTbody also need to zero
   }
   hd__syncthreads();
   #ifdef __CUDA_ARCH__
      // load in Tbody specifics in one thread
      if(threadIdx.x == 0 && threadIdx.y == 0){updateT(s_Tbody,s_cosx,s_sinx);}
      // load in dTbody specifics in another warp if needed
      if(s_dTbody != nullptr && threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1){loadTdx4(s_dTbody,s_cosx,s_sinx);}
   #else
      // load Tbody and dTbody serially
      updateT(s_Tbody,s_cosx,s_sinx);
      if(s_dTbody != nullptr){loadTdx4(s_dTbody,s_cosx,s_sinx);}
   #endif
}

template <typename T>
__host__ __device__ __forceinline__
void load_Tbdt(T *s_x, T *s_Tbody, T *d_Tbody, T *s_sinx, T *s_cosx, T *s_Tbody_dt){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // first load in Tb and dTb/dx because dTb/dt is dTb/dx*dx/dt
   load_Tb(s_x,s_Tbody,d_Tbody,s_sinx,s_cosx,s_Tbody_dt); // now dTb/dx is in s_Tbody_dt
   // then multiply wiht dx/dt note here x is just q so we can multiply by qd where s_x = [q,qd]
   #pragma unroll
   for (int body = starty; body < NUM_POS; body += dy){
      #pragma unroll
      for (int ind = startx; ind < 16; ind += dx){
         int i = body * 16 + ind;
         s_Tbody_dt[i] = s_Tbody_dt[i] * s_x[body+NUM_POS];
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void load_Tbdtdx(T *s_x, T *s_Tbody, T *d_Tbody, T *s_sinx, T *s_cosx, T *s_dTbody_dx, T *s_dTbody_dt, T *s_dTbody_dtdx){
   // first load in Tb and dTb/dx because dTb/dt is dTb/dx*dx/dt
   // then note that dTb/dtdx (because only sin and cos of x) ends up being two things
   // first dTb/dt/dq is just -Tb*qd^2 but dropping all constant terms!
   // second dTb/dt/dqd is just dTb/dq = dTb/dx
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   // compute sin/cos in parallel as well as Tbase
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
         s_sinx[ind] = sin(s_x[ind]);    
         s_cosx[ind] = cos(s_x[ind]);
   }
   // load in Tb_base and zero out dTb_dtdx
   #pragma unroll
   for (int ind = start; ind < 36*NUM_POS; ind += delta){
      s_Tbody[ind] = d_Tbody[ind];
      s_dTbody_dtdx[ind] = 0;
   }
   hd__syncthreads();
   // load in different warps or serially if on CPU
   #ifdef __CUDA_ARCH__
      if(threadIdx.x == 0 && threadIdx.y == 0){updateT(s_Tbody,s_cosx,s_sinx);}
      if(threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1){loadTdx4(s_dTbody_dx,s_cosx,s_sinx);}
      if(threadIdx.x == blockDim.x/2 - 1 && threadIdx.y == blockDim.y/2 - 1){updateT(s_dTbody_dtdx,s_cosx,s_sinx,16);}
   #else
      // load Tbody and dTbody serially
      updateT(s_Tbody,s_cosx,s_sinx);
      loadTdx4(s_dTbody_dx,s_cosx,s_sinx);
      updateT(s_dTbody_dtdx,s_cosx,s_sinx,16); // first part of dtdx is just -Tb (non constant part)

   #endif
   for (int body = starty; body < NUM_POS; body += dy){
      T vel = s_x[body+NUM_POS];
      for (int ind = startx; ind < 16; ind += dx){
         int i = body*16 + ind;  int i2 = i + 16*NUM_POS; // for qd
         s_dTbody_dt[i]    = s_dTbody_dx[i] * vel;
         s_dTbody_dtdx[i]  = -s_dTbody_dtdx[i] * vel;
         s_dTbody_dtdx[i2] = s_dTbody_dx[i];
      }
   }

}

template <typename T>
__host__ __device__ __forceinline__
void compute_T_TA_J(T *s_Tbody, T *s_Tworld, T *s_TA = nullptr, T *s_J = nullptr, T *s_TbTdt = nullptr){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   int TA_J_Flag = s_TA != nullptr && s_J != nullptr;
   // ky is now going to be the body and kx is the array val
   // compute world Ts in T (T[i] = T[i-1]*Tbody[i])
   T *Tb = s_Tbody; T *Ti = s_Tworld; T *Tim1 = s_Tworld;
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){
      #pragma unroll
      for (int ky = starty; ky < 4; ky += dy){
         #pragma unroll
         for (int kx = startx; kx < 4; kx += dx){
            // row kx of Tim1 * column ky of Tb unless body 0 then copy
            T val = 0;
            if (body == 0){
                val = Tb[ky*4+kx];
            }
            else {
               #pragma unroll
               for (int i = 0; i < 4; i++){val += Tim1[kx + 4 * i]*Tb[ky * 4 + i];}
            }
            Ti[kx + 4 * ky] = val;
            // store transpose of TL 3x3 of T in TL and BR 3x3 of s_TA for adjoint comp
            if (TA_J_Flag && kx < 3 && ky < 3){
               s_TA[body*36 + kx * 6 + ky] = val;
               s_TA[body*36 + (kx+3) * 6 + (ky+3)] = val;
            }
         }
      }
      // inc the pointers
      Tim1 = Ti; Ti += 36; Tb += 36;
      hd__syncthreads();
   }
   // check if we need to compute s_Tdt
   if (s_TbTdt != nullptr){
      // dTb/dt is in first 16*NUM_POS of s_TbTdt so place dT/dt in second 16*NUM_POS
      T *Tb_i = s_Tbody;     T *T_im1 = s_Tworld;     T *Tbdt_i = s_TbTdt;   T *Tdt_i = &s_TbTdt[16*NUM_POS];    T *Tdt_im1 = Tdt_i;
      // dT/dt[i] = dT/dt[i-1]*Tb[i] + T[i-1]*dTb/dt[i]
      #pragma unroll
      for (int body = 0; body < NUM_POS; body++){
         #pragma unroll
         for (int ky = starty; ky < 4; ky += dy){
            #pragma unroll
            for (int kx = startx; kx < 4; kx += dx){
               T val = 0;
               if (body == 0){
                  val = Tbdt_i[ky * 4 + kx];
               }
               else{
                  #pragma unroll
                  for (int i = 0; i < 4; i++){
                     val += Tdt_im1[kx + 4 * i]*Tb_i[ky * 4 + i] + T_im1[kx + 4 * i]*Tbdt_i[ky * 4 + i];
                  } 
               }
               Tdt_i[kx + 4 * ky] = val;
            }
         }
         if (body != 0){T_im1 += 36;}
         Tb_i += 36; Tbdt_i += 16; Tdt_im1 = Tdt_i; Tdt_i += 16;
         hd__syncthreads();
      }
   }
   if (!TA_J_Flag){return;}
   // compute adjoint transform of homogtransInv of T -> temp and of T but only 3rd column -> J (T in temp2 and T' stored in TL and BR of TA already)
   // since 4x4 only takes up the first 16 vals we can compute the phats in the last two 3x3 = 18 vals
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      T *phatTA = &s_Tbody[16 + 36*ind]; T *phatJ = &s_Tbody[25 + 36*ind]; Ti = &s_Tworld[36*ind];
      // for TA need to load in the result of -TA[3x3]*T[3x1_4th column]
      T tempVals0 = -(Ti[0]*Ti[12] + Ti[1]*Ti[13] + Ti[2]*Ti[14]);
      T tempVals1 = -(Ti[4]*Ti[12] + Ti[5]*Ti[13] + Ti[6]*Ti[14]); 
      T tempVals2 = -(Ti[8]*Ti[12] + Ti[9]*Ti[13] + Ti[10]*Ti[14]);
      loadAdjoint(phatTA,tempVals0,tempVals1,tempVals2);        
      // for J it is just the standard phat loading
      loadAdjoint(phatJ,&Ti[12]);
   }
   hd__syncthreads();
   // Finish TA and J by computing phat * T
   Ti = s_Tworld; T *phatTA = &s_Tbody[16]; T *phatJ = &s_Tbody[25];
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      #pragma unroll
      for (int kx = startx; kx < 9; kx += dx){
         int row = kx % 3; int column = kx / 3; T val = 0;
         // first for s_TA
         #pragma unroll
         for (int i = 0; i < 3; i++){
            val += phatTA[36*ky + row + 3 * i] * s_TA[36*ky + column * 6 + i]; // TL 3x3
         }
         s_TA[36*ky + column * 6 + (row + 3)] = val; // store in BL 3x3
         s_TA[36*ky + (column + 3) * 6 + row] = 0; // zero out TR of TA
         // then for s_J (but note only one column to compute which is 3rd column)
         if (column == 2){
            T val = 0;
            #pragma unroll
            for (int i = 0; i < 3; i++){
               val += phatJ[36*ky + row + 3 * i] * Ti[36*ky + 8 + i]; // 3rd column of T times pHat row
            }
            s_J[6*ky + row + 3] = val; // store in last 3 of J
            s_J[6*ky + row] = Ti[36*ky + 8 + row]; // load in 3rd column of T into first three
         }
      }
   }
}

// Looped such that we reduce memory from 36*NB*NB to 36*NB + 16*NB for s_dT
template <typename T>
__host__ __device__ __forceinline__
void compute_dT_dTA_dJ(T *s_Tbody, T *s_dTbody, T *s_T, T *s_dT, T *s_dTp, T *s_TA = nullptr, T *s_dTA = nullptr, T *s_dJ = nullptr){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   int TA_J_Flag = s_TA != nullptr && s_dTA != nullptr && s_dJ != nullptr;
   T *Tb = s_Tbody; T *Ti = s_T; T *TA = s_TA; T *Tim1 = s_T;
   T *phatTA = &Tb[16]; T *phatJ = &Tb[25]; T *dTb = s_dTbody;
   #pragma unroll
   for (int bodyi = 0; bodyi < NUM_POS; bodyi++){
      // compute world dTs (dT[i,j] = dT[i-1,j]*Tbody[i] + (i == j ? T[i-1]*dTbody[i] : 0))
      #pragma unroll
      for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
         T *dTij = &s_dT[36*bodyj]; T *dTim1 = &s_dTp[16*bodyj]; T *dTA = &s_dTA[36*(NUM_POS*bodyi+bodyj)];
         #pragma unroll
         for (int ind = startx; ind < 16; ind += dx){
            int ky = ind / 4; int kx = ind % 4; T val = 0;
            if (bodyi == 0){val += bodyi == bodyj ? dTb[ky * 4 + kx] : static_cast<T>(0);}
            else{
               #pragma unroll
               for (int i = 0; i < 4; i++){
                  val += dTim1[kx + 4 * i]*Tb[ky * 4 + i] + (bodyi == bodyj ? Tim1[kx + 4 * i]*dTb[ky * 4 + i] : static_cast<T>(0));
               }
            }
            dTij[kx + 4 * ky] = val;
            // store transpose of TL 3x3 of T in TL and BR 3x3 of dTA for adjoint comp
            if (TA_J_Flag && kx < 3 && ky < 3){
               dTA[kx * 6 + ky] = val; // TL
               dTA[(kx+3) * 6 + (ky+3)] = val; // BR
               dTA[(kx+3) * 6 + ky] = 0; // zero out TR of TA
            }
         }
      }
      hd__syncthreads();
      if (TA_J_Flag){
         // then also compute the derivatives of the phats note that load adjoint is simply a transformation and thus we only need to compute
         // the product rule with respect to the multiplication on tempVals -- note only need one thread per body
         for (int bodyj = start; bodyj < NUM_POS; bodyj += delta){
            T *dTij = &s_dT[36*bodyj]; T *dphatTA = &dTij[16]; T *dphatJ = &dTij[25];
            // for TA need to load in the result of -TA[3x3]*T[3x1_4th column]
            T tempVals0 = -(dTij[0]*Ti[12] + dTij[1]*Ti[13] + dTij[2]*Ti[14] + Ti[0]*dTij[12] + Ti[1]*dTij[13] + Ti[2]*dTij[14]);
            T tempVals1 = -(dTij[4]*Ti[12] + dTij[5]*Ti[13] + dTij[6]*Ti[14] + Ti[4]*dTij[12] + Ti[5]*dTij[13] + Ti[6]*dTij[14]); 
            T tempVals2 = -(dTij[8]*Ti[12] + dTij[9]*Ti[13] + dTij[10]*Ti[14] + Ti[8]*dTij[12] + Ti[9]*dTij[13] + Ti[10]*dTij[14]);
            loadAdjoint(dphatTA,tempVals0,tempVals1,tempVals2);
            // for J it is just the standard phat loading
            loadAdjoint(dphatJ,&dTij[12]);
         }
         hd__syncthreads();
         // Finish dTA and dJ by computing dphat * T + phat * dT
         #pragma unroll
         for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
            T *dTij = &s_dT[36*bodyj]; T *dphatTA = &dTij[16]; T *dphatJ = &dTij[25];
            T *dTA = &s_dTA[36*(NUM_POS*bodyi+bodyj)]; T *dJ = &s_dJ[6*(NUM_POS*bodyi+bodyj)];
            #pragma unroll
            for (int kx = startx; kx < 9; kx += dx){
               int column = kx / 3; int row = kx % 3; T val = 0;
               // first for dTA
               #pragma unroll
               for (int i = 0; i < 3; i++){
                  val += phatTA[row + 3 * i] * dTA[column * 6 + i] + dphatTA[row + 3 * i] * TA[column * 6 + i]; // TL 3x3
               }
               dTA[column * 6 + (row + 3)] = val; // store in BL 3x3
               // then for s_J (but note only one column to compute which is 3rd column)
               if (column == 2){
                  T val = 0;
                  #pragma unroll
                  for (int i = 0; i < 3; i++){
                     val += dphatJ[row + 3 * i] * Ti[8 + i] + phatJ[row + 3 * i] * dTij[8 + i]; // 3rd column of T times pHat row
                  }
                  dJ[row + 3] = val; // store in last 3 of J
                  dJ[row] = dTij[8 + row]; // load in 3rd column of dT into first three
               }
            }
         }
      }
      hd__syncthreads();
      // save down dTij into dTp for next round
      #pragma unroll
      for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
         T *dTij = &s_dT[36*bodyj]; T *dTim1 = &s_dTp[16*bodyj];
         #pragma unroll
         for (int kx = startx; kx < 16; kx += dx){dTim1[kx] = dTij[kx];}
      }
      // inc the pointers (the rest move on bodyi inc)
      Tim1 = Ti; Ti += 36; Tb += 36; dTb += 16;
      if (TA_J_Flag){TA += 36; phatTA += 36; phatJ += 36;}
      hd__syncthreads();
   }
}

// if you need the positional derivative of the time derivative of the transformation matricies (to see velocity gradients for points on body)
template <typename T>
__host__ __device__ __forceinline__
void compute_T_dtdx(T *s_Tb, T *s_Tb_dx, T *s_Tb_dt, T *s_Tb_dt_dx, T *s_T, T *s_T_dx, T *s_T_dt, T *s_T_dt_dx, T *s_T_dx_prev, T *s_T_dt_dx_prev){
    int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
    #pragma unroll
    for (int bodyi = 0; bodyi < NUM_POS; bodyi++){
        // compute d/dx[j] of dT/dt[i] = dT/dtdx[i,j] = dT/dtdx[i-1,j]*Tbody[i] + dT/dx[i-1,j]*dTbody/dt[i] + 
        //                                              (i == j ? dT/dt[i-1]*dTbody/dx[j] + T[i-1]*dTbody/dtdx[j] : 0))
        // Note1: We also need to compute T/dx[i,j] = T/dx[i-1,j]*Tbody[i] + (i == j ? T[i-1]*Tbody/dx[i] : 0))
        // Note2: The i == j comes fromt he fact that dTb/dx = 0 if i != j because only sin/cos of i vars
        // Note3: This all assumes that x[q,qd] and dqd = 0 but actually for the dqd part we still actually have something
        //        but only in the final term so we get:
        //        dT/dtdx[i,j] = dT/dtdx[i-1,j]*Tbody[i] + (i + NUM_POS == j ? T[i-1]*dTbody/dtdx[j] : 0))
        
        // The base case of body0 simplifies to just a copy or load 0
        // T/dx[0,j] = i == j ? Tbody/dx[0] : 0 and T/dtdx[0,j] = i == j ? Tbody/dtdx[0] : 0
        if (bodyi == 0){
            T *Tb_dx  = &s_Tb_dx[16*bodyi];     T *Tb_dtdx = &s_Tb_dt_dx[16*bodyi];
            #pragma unroll
            for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
                T *T_dx   = &s_T_dx[16*bodyj];  T *T_dtdx = &s_T_dt_dx[16*bodyj];
                #pragma unroll
                for (int ind = startx; ind < 16; ind += dx){
                    if (bodyi == bodyj){T_dx[ind] = Tb_dx[ind];     T_dtdx[ind] = Tb_dtdx[ind];    T_dtdx[ind+16*NUM_POS] = Tb_dtdx[ind+16*NUM_POS];}
                    else{               T_dx[ind] = 0;              T_dtdx[ind] = 0;}
                }
            }
        }
        else{
            // else start by getting pointers to i variables
            T *Tb    = &s_Tb[36*bodyi];         T *Tb_dt    = &s_Tb_dt[16*bodyi];
            T *T_im1 = &s_T[36*(bodyi-1)];      T *T_im1_dt = &s_T_dt[16*(bodyi-1)];
            #pragma unroll
            for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
                // then get pointers to ij and j variables
                T *Tb_dx  = &s_Tb_dx[16*bodyj];     T *Tb_dtdx  = &s_Tb_dt_dx[16*bodyj];
                T *T_dx   = &s_T_dx[16*bodyj];      T *T_dx_p   = &s_T_dx_prev[16*bodyj];
                T *T_dtdx = &s_T_dt_dx[16*bodyj];   T *T_dtdx_p = &s_T_dt_dx_prev[16*bodyj];
                // and for the qd pointers as well
                T *T_dtdx_qd =  &T_dtdx[16*NUM_POS];   T *T_dtdx_p_qd = &T_dtdx_p[16*NUM_POS];  T *Tb_dtdx_qd = &Tb_dtdx[16*NUM_POS];
                #pragma unroll
                for (int ind = startx; ind < 16; ind += dx){
                    // then do the matrix math
                    int ky = ind / 4; int kx = ind % 4; T val = 0; T val_dt = 0; T val_dt_qd = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++){
                        // get inds for row*col of matricies
                        int ind1 = kx + 4 * i;  int ind2 = ky * 4 + i;
                        // index into appropriate matricies
                        val        += T_dx_p[ind1]      * Tb[ind2];
                        val_dt     += T_dtdx_p[ind1]    * Tb[ind2] + T_dx_p[ind1] * Tb_dt[ind2];
                        val_dt_qd  += T_dtdx_p_qd[ind1] * Tb[ind2];
                        if (bodyi == bodyj){
                            val       += T_im1[ind1]    * Tb_dx[ind2];
                            val_dt    += T_im1_dt[ind1] * Tb_dx[ind2] + T_im1[ind1] * Tb_dtdx[ind2];
                            val_dt_qd += T_im1[ind1]    * Tb_dtdx_qd[ind2];
                        }
                    }
                    T_dx[ind]      = val;
                    T_dtdx[ind]    = val_dt;
                    T_dtdx_qd[ind] = val_dt_qd;
                }
            }
        }
        hd__syncthreads();
        // save down T_dt_dx and T_dx into T_dt_dx_prev and T_dx_prev
        #pragma unroll
        for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
            T *T_dx      = &s_T_dx[16*bodyj];      T *T_dx_p      = &s_T_dx_prev[16*bodyj];
            T *T_dtdx    = &s_T_dt_dx[16*bodyj];   T *T_dtdx_p    = &s_T_dt_dx_prev[16*bodyj];
            T *T_dtdx_qd = &T_dtdx[16*NUM_POS];    T *T_dtdx_p_qd = &T_dtdx_p[16*NUM_POS];
            #pragma unroll
            for (int kx = startx; kx < 16; kx += dx){
               T_dx_p[kx]      = T_dx[kx];  
               T_dtdx_p[kx]    = T_dtdx[kx];
               T_dtdx_p_qd[kx] = T_dtdx_qd[kx];
            }
        }
        hd__syncthreads();
    }
}

// compute Iw, Icrbs, twsits, and dIw if needed uses TA,J,x, and temp space as well as dTA if dIw is needed
template <typename T>
__host__ __device__ __forceinline__
void compute_Iw_Icrbs_twist(T *s_I, T *s_Icrbs, T *s_twist, T *s_TA, T *s_J, T *s_x, T *s_temp, T *s_dTA = nullptr, T *s_temp2 = nullptr){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   // I = I*TA to start the inertia comp -- store in temp
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // row L * column R stores in (r,c)
      #pragma unroll
      for (int kx = startx; kx < 36; kx += dx){
         int r = kx % 6;
         int c = kx / 6;
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_I[36*ky + r + 6 * i] * s_TA[36*ky + c * 6 + i];
         }
         s_temp[36*ky + c * 6 + r] = val; 
      }
   }
   hd__syncthreads();
   // compute dIw if needed
   if (s_dTA != nullptr){
      // we are now going to start running into memory issues so we are going to loop NUM_POS times
      // lets use Icrbs and temp as the temp space and load the finals in temp2 and then back into dTA to save space
      // dIW = dTA'*(I*TA) + TA'*(I*dTA) -- note I*TA is already in s_temp
      // 
      #pragma unroll
      for (int bodyi = 0; bodyi < NUM_POS; bodyi++){
         #pragma unroll
         for (int ky = starty; ky < NUM_POS; ky += dy){
            #pragma unroll
            for (int kx = startx; kx < 36; kx += dx){
               int r = kx % 6;
               int c = kx / 6;
               T val = 0;
               #pragma unroll
               for (int i = 0; i < 6; i++){
                  val += s_I[36*bodyi + r + 6 * i] * s_dTA[36*(bodyi*NUM_POS+ky) + c * 6 + i];
               }
               s_Icrbs[36*ky + c * 6 + r] = val;   
            }
         }
         hd__syncthreads();
         #pragma unroll
         for (int ky = starty; ky < NUM_POS; ky += dy){
            #pragma unroll
            for (int kx = startx; kx < 36; kx += dx){
               T val = 0;
               int r = kx % 6;
               int c = kx / 6;
               #pragma unroll
               for (int i = 0; i < 6; i++){
                  val += s_dTA[36*(bodyi*NUM_POS+ky) + r * 6 + i] * s_temp[36*bodyi + c * 6 + i];
                  val += s_TA[36*bodyi + r * 6 + i] * s_Icrbs[36*ky + c * 6 + i];
               }
               s_temp2[36*ky + c*6+r] = val; // load this bodyi into temp2 
            }
         }
         hd__syncthreads();
         //T *s_dIw = &s_dTA[36*bodyi*NUM_POS];
         #pragma unroll
         for (int ky = starty; ky < NUM_POS; ky += dy){
            #pragma unroll
            for (int kx = startx; kx < 36; kx += dx){
               s_dTA[36*(bodyi*NUM_POS+ky) + kx] = s_temp2[36*ky+kx]; // then copy over into s_dTA
            }
         }
         // no sync needed because can start computing next step from next bodyi without incident
      }
   }
   // IW = TA'*(I*TA) to finish the inertia comp which we can now safely overwrite s_I
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // row L * column R stores in (r,c)
      #pragma unroll
      for (int kx = startx; kx < 36; kx += dx){
         int r = kx % 6;
         int c = kx / 6;
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_TA[36*ky + r * 6 + i] * s_temp[36*ky + c * 6 + i];
         }
         s_I[36*ky + c * 6 + r] = val; 
      }
   }
   hd__syncthreads();
   // compute the recursion for the CRBI which is just a summation of the world Is
   // and finish the twist recursion which is also a summation
   // sum IW to IC IC[i] = Sum_j>i IC[j] matrix 6x6 (avoids syncthreads and only NUM_POS wasted additions is probs faster)
   #pragma unroll
   for (int ind = start; ind < 36; ind += delta){
      T val = 0;
      #pragma unroll
      for (int body = NUM_POS-1; body >= 0; body--){
         val += s_I[36*body + ind];
         s_Icrbs[36*body + ind] = val;
         // and clear Temp and TA for later
         s_TA[36*body + ind] = 0;
         s_temp[36*body + ind] = 0;
      }
   }
   // compute the twist recursion -- twist[i] = SUM_j<=i twist[j] where (twists[j] = J[j]*x[nbodies+j])
   #pragma unroll
   for (int ind = start; ind < 6; ind += delta){
      #pragma unroll
      for (int body = 0; body < NUM_POS; body++){
         s_twist[6*body + ind] = s_J[6*body + ind] * s_x[NUM_POS+body] + (body > 0 ? s_twist[6*(body-1) + ind] : static_cast<T>(0));
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_JdotV(T *s_JdotV, T *s_twist, T *s_J, T *s_x, T *s_temp){
   int start, delta; singleLoopVals(&start,&delta);
   // compute the CRMs of the twists -- temp needs to have been cleared earlier
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfm(&s_temp[36*ind],&s_twist[6*ind],0);
   }
   hd__syncthreads();   
   // now compute JdotV[i] = JdotV[i-1] + crms[i]*J[i]*s_x[NUM_POS + i] and store in s_JdotV
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){
      #pragma unroll
      for (int ind = start; ind < 6; ind += delta){
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_temp[36*body + ind + i*6]*s_J[6*body + i];
         }
         s_JdotV[6*body + ind] = s_x[NUM_POS + body]*val + (body > 0 ? s_JdotV[6*(body-1) + ind] : static_cast<T>(0));
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dtwist(T *s_dTwist, T *s_J, T *s_dJ, T *s_x){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // dtwist[i,j] (NB*NB*6*2 b/c qd) = dJ[i,j]*qd[i] + J[i]*dqd[i,j] + dtwist[i-1,j]
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // first do the dqs where we only have dJ[i,j]*qd[i] + dtwist[i-1,j]
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = s_dJ[6*(body*NUM_POS + ky) + kx]*s_x[NUM_POS + body];
            if (body > 0){
               val += s_dTwist[6*((body-1)*2*NUM_POS + ky) + kx];
            }
            s_dTwist[6*(body*2*NUM_POS + ky) + kx] = val;
         }
      }
      // then the dqds where we only have J[i]*dqd[i,j] + dtwist[i-1,j]
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = ky == body ? s_J[6*body + kx] : static_cast<T>(0);
            if (body > 0){
               val += s_dTwist[6*((body-1)*2*NUM_POS + NUM_POS + ky) + kx];
            }
            s_dTwist[6*(body*2*NUM_POS + NUM_POS + ky) + kx] = val;
         }
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dJdotV(T *s_dJdotV, T *s_twist, T *s_dTwist, T *s_J, T *s_dJ, T *s_x, T *s_temp, T *s_temp2){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   // dJdotV[i] (NB*NB*6*2 b/c qd) = crm(dtwist)*J*qd + crm(twist)*(dJ*qd + J*dqd) + dJdotV[i-1] 
   // first form the crms in s_temp and s_temp2 (need two passes b/c it assume each is 36*NB and we need 36*NB*2 for each)
   // then mulitply out to get the answer
   // first lets load in the crm(twist) into s_temp2 -- note we need to zero it all first so use the zeroing crfmz b/c the constant syncing
   // to rezero will probably be slower
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfmz(&s_temp2[36*ind],&s_twist[6*ind],0);
   }
   // then we loop by body forming the dwtists and the results
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // first form the dcrms for the qs
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp[36*ind],&s_dTwist[6*(body*2*NUM_POS + ind)],0);
      }
      hd__syncthreads();
      // then multiply for the qs so dqd = 0 thus we need = (crm(dtwist)*J + crm(twist)*dJ)*qd + dJdotV[i-1]
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = 0;
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += s_temp[36*ky + kx + 6 * i]*s_J[6*body + i] + s_temp2[36*body + kx + 6 * i]*s_dJ[6*(body*NUM_POS + ky) + i];
            }
            val *= s_x[NUM_POS + body];
            if (body > 0){
               val += s_dJdotV[6*((body-1)*2*NUM_POS + ky) + kx];
            }
            s_dJdotV[6*(body*2*NUM_POS + ky) + kx] = val;
         }
      }
      hd__syncthreads();
      // then form the crms for the qds but note that the twists dont change its just the dtwists
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp[36*ind],&s_dTwist[6*(body*2*NUM_POS + NUM_POS + ind)],0);
      }
      hd__syncthreads();
      // then multiply for the qds so dJ = 0 thus we need = (crm(dtwist)*qd + crm(twist)*dqd)*J + dJdotV[i-1] 
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = 0;
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += (s_temp[36*ky + kx + 6 * i]*s_x[NUM_POS + body] + (ky == body ? s_temp2[36*body + kx + 6 * i] : static_cast<T>(0)))*s_J[6*body + i];
            }
            if (body > 0){
               val += s_dJdotV[6*((body-1)*2*NUM_POS + NUM_POS + ky) + kx];
            }
            s_dJdotV[6*(body*2*NUM_POS + NUM_POS + ky) + kx] = val;
         }
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_M_Tau(T *s_M, T *s_Tau, T *s_W, T *s_JdotV, T *s_F, T *s_Icrbs, T *s_twist, T *s_J, T *s_I, T *s_x, T *s_u, T *s_temp, T *s_temp2, T *s_TA){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   // compute sub parts for wrenches and the force matrix for the mass matrix comp
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // compute wrench subparts
      #pragma unroll
      for (int kx = startx; kx < 6; kx += dx){
         // temp_c1 = I_world * W_twist (holding the twist)
         // temp_c2 = I_world * grav + JdV
         // form the force matrices for the mass matrix comp (F = I_crbs[i]*J[i])
         T val = 0;
         T val2 = 0;
         T val3 = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            int Iind = 36*ky + kx + i * 6;
            val  += s_I[Iind] * s_twist[6*ky + i];
            val2 += s_I[Iind] * (s_JdotV[6*ky + i] + static_cast<T>(i == 5 ? GRAVITY : 0)); //for arm gravity vec is [0 0 0 0 0 g]
            val3 += s_Icrbs[Iind] * s_J[6*ky + i];
         }
         s_temp[36*ky + kx] = val;        
         s_temp[36*ky + 6 + kx] = val2;
         s_F[ky*6 + kx] = val3;
      }
      // finally form the crfs for the wrenches (TA cleared earlier)
      int flag = 1; // optimized away in cpu case
      #ifdef __CUDA_ARCH__
         flag = threadIdx.x == blockDim.x - 1; // use last thread b/c less likely to be looping above
      #endif
      if (flag){crfm(&s_TA[36*ky],&s_twist[6*ky],1);}
   }
   hd__syncthreads();
   // W[i] and mass matrix
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // compute W = crf(W_twist) * IC * W_twist + IC*(g + JdV) = <<<TA*temp_c1 + temp_c2 = W>>>
      #pragma unroll
      for (int kx = startx; kx < 6; kx += dx){
         T val = 0;
         for (int i = 0; i < 6; i++){
            val += s_TA[36*ky + kx + i * 6]*s_temp[36*ky + i];
         }
         s_W[6*ky + kx] = val + s_temp[36*ky + 6 + kx];
         //printf("Body[%d]_[%d]: W[%f] = IgJdV[%f] + crf*IW[%f]\n",ky,kx,s_W[kx],s_temp[36*ky + 6 + kx],val);
      }
      // and at the same time the Mass matrix which we store back in s_temp2
      #pragma unroll
      for (int kx = startx; kx < NUM_POS; kx += dx){
         // M(i,j<=1) = M(j<=1,i) = J[j]*F[i]
         int jInd, iInd;
         if (kx <= ky){
            jInd = kx;
            iInd = ky;
         }
         else{
            jInd = ky;
            iInd = kx;
         }
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_J[6*jInd + i] * s_F[6*iInd + i];
         }
         s_M[ky*NUM_POS + kx] = val;
         // also load in an identity next to it to prep for inverse
         s_M[(ky+NUM_POS)*NUM_POS + kx] = static_cast<T>(kx == ky ? 1 : 0);
      }
   }
   hd__syncthreads();
   // net W: sum net wrenches W[i] = Sum_j>=i W[j] vector 6x1
   #pragma unroll
   for (int ind = start; ind < 6; ind += delta){
      T val = 0;
      #pragma unroll
      for (int body = NUM_POS - 1; body >= 0; body--){
         val += s_W[6*body + ind];
         s_W[6*body + ind] = val;
      }
   }
   hd__syncthreads();
   // compute the bias force
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      // C(i) = W(i)*J(i) -- store in end of temp2 (also subtract from u for later we are forming tau)
      T val = 0;
      for (int i = 0; i < 6; i++){
         val += s_J[6*ind + i] * s_W[6*ind + i];
      }     
      // for our robot damping is all velocity dependent and =0.5v and B = I so tau = u-(c+0.5v)
      s_Tau[ind] = s_u[ind] - (val + static_cast<T>(0.5)*s_x[NUM_POS+ind]);
      // printf("Ind[%d]: Tau[%f] = u[%f] - (val[%f] + 0.5qd[%f])\n",ind,s_Tau[ind],s_u[ind],val,0.5*s_x[NUM_POS+ind]);
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dWb(T *s_dWb, T *s_JdotV, T *s_dJdotV, T *s_twist, T *s_dTwist, T *s_Iw, T *s_dIw, T *s_temp, T *s_temp2, T *s_temp3){
   int start, delta; singleLoopVals(&start,&delta);
   // dWb (NB*NB*6) = dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
   // again first form the crfs in s_temp and s_temp2 (need two passes b/c it is assumed 36*NB and we need 36*NB*2 for each)
   // then mulitply out to get the answer -- note s_temp3 only needs to be 6*NB in size because storing temp totals
   // first form the crfs
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfmz(&s_temp[36*ind],&s_twist[6*ind],1);
   }
   hd__syncthreads();
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // fist form the dcrfs for the qs
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp2[36*ind],&s_dTwist[6*(body*2*NUM_POS + ind)],1);
      }
      hd__syncthreads();
      // the multiply dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
      // our issue here is that we actually need far more space than we have so we are going to loop this again unfortunately
      #pragma unroll
      for (int dbody = 0; dbody < NUM_POS; dbody++){ // derivative body
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val0 = 0; //dIw*([0--9.81] + JdotV) + Iw*dJdotV
            T val1 = 0; //crf(dtwist)*Iw*twist
            T val2 = 0; //crf(twist)*(dIw*twist + Iw*dtwist)
            #pragma unroll
            for (int i = 0; i < 6; i++){
               T Iw = s_Iw[36*body + ind + 6 * i];
               T dIw = s_dIw[36*(body*NUM_POS + dbody) + ind + 6 * i];
               T tw = s_twist[6*body + i];
               T dtw = s_dTwist[6*(body*2*NUM_POS + dbody) + i];
               T dJdV = s_dJdotV[6*(body*2*NUM_POS + dbody) + i];
               val0 += dIw*(s_JdotV[6*body + i] + static_cast<T>(i == 5 ? GRAVITY : 0)) + Iw*dJdV;
               val1 += Iw*tw;
               val2 += dIw*tw + Iw*dtw;
            }
            // store the temp vals in s_temp3
            s_temp3[3*ind] = val0;
            s_temp3[3*ind + 1] = val1;
            s_temp3[3*ind + 2] = val2;
         }
         hd__syncthreads();
         // now finish it off with val0 + crf(dtwist)*val1 + crf(twist)*val2
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val = s_temp3[3*ind];
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += s_temp2[36*dbody + ind + 6 * i]*s_temp3[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp3[3*i + 2];
            }
            s_dWb[6*(body*2*NUM_POS + dbody) + ind] = val;
         }
         hd__syncthreads();
      }
      // then form the crfs for the qds and again note that only the dtwist changes
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp2[36*ind],&s_dTwist[6*(body*2*NUM_POS + NUM_POS + ind)],1);
      }
      hd__syncthreads();
      // then multiply again
      #pragma unroll
      for (int dbody = 0; dbody < NUM_POS; dbody++){ // derivative body
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            // note now that dIw == 0 so can drop those terms
            T val0 = 0; //Iw*dJdotV
            T val1 = 0; //crf(dtwist)*Iw*twist
            T val2 = 0; //crf(twist)*Iw*dtwist
            #pragma unroll
            for (int i = 0; i < 6; i++){
               T Iw = s_Iw[36*body + ind + 6 * i];
               T tw = s_twist[6*body + i];
               T dtw = s_dTwist[6*(body*2*NUM_POS + NUM_POS + dbody) + i];
               T dJdV = s_dJdotV[6*(body*2*NUM_POS + NUM_POS + dbody) + i];
               val0 += Iw*dJdV;
               val1 += Iw*tw;
               val2 += Iw*dtw;
            }
            // store the temp vals in s_temp3
            s_temp3[3*ind] = val0;
            s_temp3[3*ind + 1] = val1;
            s_temp3[3*ind + 2] = val2;
         }
         hd__syncthreads();
         // now finish it off with val0 + crf(dtwist)*val1 + crf(twist)*val2
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val = s_temp3[3*ind];
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += s_temp2[36*dbody + ind + 6 * i]*s_temp3[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp3[3*i + 2];
            }
            s_dWb[6*(body*2*NUM_POS + NUM_POS + dbody) + ind] = val;
         }
         hd__syncthreads();
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dTau(T *s_dTau, T *s_dWb, T *s_W, T *s_J, T *s_dJ){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // dTau = -dC for the arm and dC (NB*NB) = dJ*W + J*SUM(dWb) + 0.5dqd(aka eye) -- note dJ only exists for qs
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // main body
      #pragma unroll
      for (int kx = startx; kx < 2*NUM_POS; kx += dx){ // derivative body
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            T dW = 0;
            #pragma unroll
            for (int j = ky; j < NUM_POS; j++){
               dW += s_dWb[6*(j*2*NUM_POS + kx) + i];
            }
            val += (kx < NUM_POS ? s_dJ[6*(ky*NUM_POS + kx) + i]*s_W[6*ky + i] : static_cast<T>(0)) + s_J[6*ky + i]*dW;
         }
         s_dTau[kx*NUM_POS + ky] = -(val + static_cast<T>(kx - NUM_POS == ky ? 0.5 : 0));
      }
   }
}

// Looped version of comptueDTau for memory efficiency so we only need current and parent of each body since serial chain
// so that means NB*6*2 for (dtwist, dtwistp, dJdotV, dJdotVp, dWb, dWp, dTau)
// temp = 36*NB, temp2 = 36*NB, temp3 = 36*NB, temp4 = 6*NB;
template <typename T>
__host__ __device__ __forceinline__
void compute_dTau(T *s_dTau, T *s_W, T *s_dWb, T *s_dWp, T *s_JdotV, T *s_dJdotV, T *s_dJdotVp, T *s_twist, T *s_dTwist, T *s_dTwistp, T *s_Iw, T *s_dIw, T *s_J, T *s_dJ, T *s_x, T *s_temp, T *s_temp2, T *s_temp3, T *s_temp4){
   int start, delta; singleLoopVals(&start,&delta);
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // we loop over bodies for memory efficiency but also pre-compute some terms that don't change
   // first clear the prev variables and temp vars for the crfms
   #pragma unroll
   for (int ind = start; ind < 36*NUM_POS; ind += delta){
      s_temp[ind] = 0;
      s_temp2[ind] = 0;
      s_temp3[ind] = 0;
      if (ind < 12*NUM_POS){
         s_dWp[ind] = 0;
         s_dJdotVp[ind] = 0;
         s_dTwistp[ind] = 0;
      }
   }
   hd__syncthreads();
   // first up crm(twist) into s_temp
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfm(&s_temp[36*ind],&s_twist[6*ind],0);
   }
   // then start the loops
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // dTwist[i,j] (NB*6*2 b/c qd) = dJ[i,j]*qd[i] + J[i]*dqd[i,j] + dtwist[i-1,j]
      // first do the dqs where we only have dJ[i,j]*qd[i] + dtwist[i-1,j]
      // then the dqds where we only have J[i]*dqd[i,j] + dtwist[i-1,j]
      #pragma unroll
      for (int ky = starty; ky < 2*NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val;
            if (body < NUM_POS){val = s_dJ[6*(body*NUM_POS + ky) + kx]*s_x[NUM_POS + body];}
            else{val = ky == body ? s_J[6*body + kx] : static_cast<T>(0);}
            if (body > 0){val += s_dTwistp[6*ky + kx];
            }
            s_dTwist[6*ky + kx] = val;
         }
      }
      hd__syncthreads();
      // dJdotV[i] (NB*6*2 b/c qd) = crm(dtwist)*J*qd + crm(twist)*(dJ*qd + J*dqd) + dJdotV[i-1] 
      // first form the dcrms for the qs and qds
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS; ind += delta){
         if (ind < NUM_POS){
            crfm(&s_temp2[36*ind],&s_dTwist[6*ind],0);   
         }
         else{
            crfm(&s_temp3[36*(ind-NUM_POS)],&s_dTwist[6*ind],0);
         }
      }
      hd__syncthreads();
      // then multiply for the qs so dqd = 0 thus we need = (crm(dtwist)*J + crm(twist)*dJ)*qd + dJdotV[i-1]
      // then multiply for the qds so dJ = 0 thus we need = (crm(dtwist)*qd + crm(twist)*dqd)*J + dJdotV[i-1] 
      #pragma unroll
      for (int ky = starty; ky < 2*NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = 0;
            #pragma unroll
            for (int i = 0; i < 6; i++){
               if (ky < NUM_POS){
                  val += (s_temp2[36*ky + kx + 6 * i]*s_J[6*body + i] + s_temp[36*body + kx + 6 * i]*s_dJ[6*(body*NUM_POS + ky) + i])*s_x[NUM_POS + body];
               }
               else{
                  val += (s_temp3[36*ky + kx + 6 * i]*s_x[NUM_POS + body] + (ky == body ? s_temp[36*body + kx + 6 * i] : static_cast<T>(0)))*s_J[6*body + i];
               }
            }
            if (body > 0){val += s_dJdotVp[6*ky + kx];}
            s_dJdotV[6*ky + kx] = val;
         }
      }
      hd__syncthreads();
      // dWb (NB*NB*6) = dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
       // first form the dcrms for the qs and qds
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS; ind += delta){
         if (ind < NUM_POS){
            crfm(&s_temp2[36*ind],&s_dTwist[6*ind],1);   
         }
         else{
            crfm(&s_temp3[36*(ind-NUM_POS)],&s_dTwist[6*ind],1);
         }
      }
      hd__syncthreads();
      // the multiply dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
      // our issue here is that we actually need far more space than we have so we are going to loop this again unfortunately
      #pragma unroll
      for (int dbody = 0; dbody < 2*NUM_POS; dbody++){ // derivative body
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            // note: for dbody > NUM_POS then dIw = 0 so drop those terms
            T val0 = 0; //dIw*([0--9.81] + JdotV) + Iw*dJdotV
            T val1 = 0; //crf(dtwist)*Iw*twist
            T val2 = 0; //crf(twist)*(dIw*twist + Iw*dtwist)
            #pragma unroll
            for (int i = 0; i < 6; i++){
               T Iw = s_Iw[36*body + ind + 6 * i];
               
               T tw = s_twist[6*body + i];
               T dtw = s_dTwist[6*(body*2*NUM_POS + dbody) + i];
               T dJdV = s_dJdotV[6*(body*2*NUM_POS + dbody) + i];
               if (dbody < NUM_POS){
                  T dIw = s_dIw[36*(body*NUM_POS + dbody) + ind + 6 * i];   
                  val0 += dIw*(s_JdotV[6*body + i] + static_cast<T>(i == 5 ? GRAVITY : 0));
                  val2 += dIw*tw;
               }
               val0 += Iw*dJdV;
               val1 += Iw*tw;
               val2 += Iw*dtw;
            }
            // store the temp vals in s_temp4
            s_temp4[3*ind] = val0;
            s_temp4[3*ind + 1] = val1;
            s_temp4[3*ind + 2] = val2;
         }
         hd__syncthreads();
         // now finish it off with val0 + crf(dtwist)*val1 + crf(twist)*val2
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val = s_temp4[3*ind];
            #pragma unroll
            for (int i = 0; i < 6; i++){
               if (ind < NUM_POS){
                  val += s_temp2[36*dbody + ind + 6 * i]*s_temp4[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp4[3*i + 2];   
               }
               else{
                  val += s_temp3[36*(dbody-NUM_POS) + ind + 6 * i]*s_temp4[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp4[3*i + 2];
               }
            }
            s_dWb[6*dbody + ind] = val;
         }
         hd__syncthreads();
      }
      // dTau = -dC for the arm and dC (NB*NB) = dJ*W + J*SUM(dWb) + 0.5dqd(aka eye) -- note dJ only exists for qs
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS; ind += delta){
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            T dW = s_dWb[6*ind + i] + s_dWp[6*ind + i]; // sum for W
            val += (ind < NUM_POS ? s_dJ[6*(body*NUM_POS + ind) + i]*s_W[6*body + i] : static_cast<T>(0)) + s_J[6*body + i]*dW;
         }
         s_dTau[ind*NUM_POS + body] = -(val + static_cast<T>(ind - NUM_POS == body ? 0.5 : 0));
      }
      hd__syncthreads();
      // now save current into prev for next round
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS*6; ind += delta){
         s_dWp[ind] = s_dWb[ind];
         s_dJdotVp[ind] = s_JdotV[ind];
         s_dTwistp[ind] = s_dTwist[ind];
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_qdd(T *s_qdd, T *s_Minv, T *s_Tau){
   int start, delta; singleLoopVals(&start,&delta);
   // for the arm B = I so qdd = Hinv*tau
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      T val = 0;
      for (int i = 0; i < NUM_POS; i++){
         val += s_Minv[ind + NUM_POS*i] * s_Tau[i];
      }     
      s_qdd[ind] = val;
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dM(T *s_dM, T *s_Icrbs, T *s_dIw, T *s_J, T *s_dJ, T *s_F, T *s_temp){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // dM[i,j]/dk = dJ[j,k]*Icrbs[i]*J[i] + J[j]*(dIcrbs[i,k]*J[i] + Icrbs[i]*dJ[i,k])
   // so we need s_F[i] = s_Icrbs[i]*s_J[i]
   //            s_temp[i,k] = dIcrbs[i,k]*J[i] + Icrbs[i]*dJ[i,k]
   // note that we only have dIw and dIcrbs = SUM_j>i dIw[i] so need to dynamically sum
   #pragma unroll
   for (int bodyi = starty; bodyi < NUM_POS; bodyi += dy){
      for (int kx = startx; kx < NUM_POS*6; kx += dx){
         int bodyk = kx / 6;
         int r = kx % 6;
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            // need to sum up the dIw for the dIcrbs
            T dIcrbs = 0;
            #pragma unroll
            for (int j = bodyi; j < NUM_POS; j++){
               dIcrbs += s_dIw[36*(j*NUM_POS + bodyk) + r + 6 * i];
            }
            //dIcrbs[i,k]*J[i] + Icrbs[i]*dJ[i,k]
            val += dIcrbs*s_J[6*bodyi + i] + s_Icrbs[36*bodyi + r + 6 * i]*s_dJ[6*(bodyi*NUM_POS + bodyk) + i];
         }
         s_temp[6*(bodyi*NUM_POS + bodyk) + r] = val;
      }
      int reps = 0;
      #ifdef __CUDA_ARCH__
         if(threadIdx.x >= blockDim.x - 6){reps = 1;} // possibly in separate warp
      #else
         reps = 6;
      #endif
      #pragma unroll
      for(int rep = 0; rep < reps; rep++){
         #ifdef __CUDA_ARCH__
            int r = threadIdx.x % 6;
         #else
            int r = rep;
         #endif
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){val += s_Icrbs[36*bodyi + r + 6 * i]*s_J[6*bodyi + i];}
         s_F[bodyi*6 + r] = val;
      }
   }
   hd__syncthreads();
   // now dM[i,j]/dk = dJ[j,k]*F[i] + J[j]*temp[i][k] --> store in s_dM for now
   #pragma unroll
   for (int bodyk = starty; bodyk < NUM_POS; bodyk += dy){
      #pragma unroll
      for (int kx = startx; kx < NUM_POS*NUM_POS; kx += dx){
         int r = kx % NUM_POS;
         int c = kx / NUM_POS;
         int jInd, iInd;
         if (r <= c){
            jInd = r;
            iInd = c;
         }
         else{
            jInd = c;
            iInd = r;
         }
         T val = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_dJ[6*(jInd*NUM_POS + bodyk) + i] * s_F[6*iInd + i] + s_J[6*jInd + i] * s_temp[6*(iInd*NUM_POS + bodyk) + i];
         }
         s_dM[NUM_POS*NUM_POS*bodyk + c * NUM_POS + r] = val;
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dqdd_dM(T *s_dqdd, T *s_dM, T *s_Minv, T *s_qdd, T *s_temp){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // then compute first half of the compute for dqdd (-Minv^T*dM*Minv*tau = -Minv^T*dM*qdd)
   // note dqdd_dx will be 2*NUM_POS*NUM_POS but dM part is only half of the first NUM_POS*NUM_POS part
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // pick the dx
      #pragma unroll
      for (int kx = startx; kx < NUM_POS; kx += dx){ // pick the row output
         T *dM = &s_dM[NUM_POS*NUM_POS*ky + kx];
         // then dM(stride by k)*qdd in tspace
         T val = 0;
         #pragma unroll
         for (int i = 0; i < NUM_POS; i++){
            val += dM[i*NUM_POS]*s_qdd[i];
         }
         s_temp[ky*NUM_POS + kx] = val;
      }
   }
   hd__syncthreads();
   // now we need -Minv^T*tspace -> dqdd
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // pick the dx
      #pragma unroll
      for (int kx = startx; kx < NUM_POS; kx += dx){ // pick the row output
         T val = 0;
         #pragma unroll
         for (int i = 0; i < NUM_POS; i++){
            val += s_Minv[kx*NUM_POS + i]*s_temp[ky*NUM_POS + i];
         }
         s_dqdd[ky*NUM_POS + kx] = -val;
         s_dqdd[(ky+NUM_POS)*NUM_POS + kx] = 0; // set dqd part to 0 for now
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void finish_dqdd(T *s_dqdd, T *s_dTau, T *s_Minv){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   T *s_dqdd_du = &s_dqdd[2*NUM_POS*NUM_POS];
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // row of invH
      #pragma unroll
      for (int kx = startx; kx < 2*NUM_POS; kx += dx){ // col of dTau
         T val = 0;
         #pragma unroll
         for (int i = 0; i < NUM_POS; i++){
            val += s_Minv[ky + NUM_POS * i] * s_dTau[kx*NUM_POS + i];
         }
         s_dqdd[kx*NUM_POS + ky] += val;
         // also load in dqdd_du which for arm is just Hinv
         if (kx < NUM_POS){s_dqdd_du[kx*NUM_POS + ky] = s_Minv[kx*NUM_POS + ky];}
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_eePos(T *s_T, T *s_eePos, T *s_dT = nullptr, T *s_deePos = nullptr, T *s_temp = nullptr){
   int start, delta; singleLoopVals(&start,&delta);
   #ifdef __CUDA_ARCH__
      bool thread0Flag = threadIdx.x == 0 && threadIdx.y == 0;
   #else
      bool thread0Flag = 1;
   #endif
   T *Tee = &s_T[(NUM_POS-1)*36];
   // get hand pos and factors for multiplication in one thread
   if (thread0Flag){
      s_eePos[0] = Tee[0]*static_cast<T>(EE_ON_LINK_X) + Tee[4]*static_cast<T>(EE_ON_LINK_Y) + Tee[8]*static_cast<T>(EE_ON_LINK_Z)  + Tee[12];
      s_eePos[1] = Tee[1]*static_cast<T>(EE_ON_LINK_X) + Tee[5]*static_cast<T>(EE_ON_LINK_Y) + Tee[9]*static_cast<T>(EE_ON_LINK_Z)  + Tee[13];
      s_eePos[2] = Tee[2]*static_cast<T>(EE_ON_LINK_X) + Tee[6]*static_cast<T>(EE_ON_LINK_Y) + Tee[10]*static_cast<T>(EE_ON_LINK_Z) + Tee[14];
      s_eePos[3] = (T) atan2(Tee[6],Tee[10]);
      s_eePos[4] = (T) atan2(-Tee[2],sqrt(Tee[6]*Tee[6]+Tee[10]*Tee[10]));
      s_eePos[5] = (T) atan2(Tee[1],Tee[0]);
   }
   // if computing derivatives first compute factors in temp memory
   bool dFlag = (s_dT != nullptr) && (s_deePos != nullptr) && (s_temp != nullptr);
   if (dFlag){
      if (thread0Flag){
         T factor3 = Tee[6]*Tee[6] + Tee[10]*Tee[10];
         T factor4 = static_cast<T>(1)/(Tee[2]*Tee[2] + factor3);
         T factor5 = static_cast<T>(1)/(Tee[1]*Tee[1] + Tee[0]*Tee[0]);
         T sqrtfactor3 = (T) sqrt(factor3);
         s_temp[0] = -Tee[6]/factor3;
         s_temp[1] = Tee[10]/factor3;
         s_temp[2] = Tee[2]*Tee[6]*factor4/sqrtfactor3;
         s_temp[3] = Tee[2]*Tee[10]*factor4/sqrtfactor3;
         s_temp[4] = -sqrtfactor3*factor4;
         s_temp[5] = -Tee[1]*factor5;
         s_temp[6] = Tee[0]*factor5;
      }
      hd__syncthreads();
      // then compute all dk in parallel (note dqd is 0 so only dq needed)
      #pragma unroll
      for (int k = start; k < NUM_POS; k += delta){
         T *dT = &s_dT[36*k]; // looped variant only saves final dT[i]/dx but thats all we need (which is why that is ok)
         s_deePos[k*6]     = dT[0]*static_cast<T>(EE_ON_LINK_X) + dT[4]*static_cast<T>(EE_ON_LINK_Y) + dT[8]*static_cast<T>(EE_ON_LINK_Z)  + dT[12];
         s_deePos[k*6 + 1] = dT[1]*static_cast<T>(EE_ON_LINK_X) + dT[5]*static_cast<T>(EE_ON_LINK_Y) + dT[9]*static_cast<T>(EE_ON_LINK_Z)  + dT[13];
         s_deePos[k*6 + 2] = dT[2]*static_cast<T>(EE_ON_LINK_X) + dT[6]*static_cast<T>(EE_ON_LINK_Y) + dT[10]*static_cast<T>(EE_ON_LINK_Z) + dT[14];
         s_deePos[k*6 + 3] = s_temp[0]*dT[10] + s_temp[1]*dT[6];
         s_deePos[k*6 + 4] = s_temp[2]*dT[6]  + s_temp[3]*dT[10] + s_temp[4]*dT[2];
         s_deePos[k*6 + 5] = s_temp[5]*dT[0]  + s_temp[6]*dT[1];
      }
   }
}
template <typename T>
__host__ __device__ __forceinline__
void compute_eePos(T *s_eePos, T *s_T, T *s_Tb, T *s_sinq, T *s_cosq, T *s_x, T *d_Tbody){
   load_Tb(s_x,s_Tb,d_Tbody,s_cosq,s_sinq);
   hd__syncthreads();
   // then compute Tbody -> T
   compute_T_TA_J(s_Tb,s_T);
   hd__syncthreads();
   //compute the hand position
   compute_eePos(s_T,s_eePos);
}
template <typename T>
__host__ __device__ __forceinline__
void compute_eePos(T *s_T, T *s_eePos, T *s_dT, T *s_deePos, T *s_sinq, 
                   T *s_Tb, T *s_dTb, T *s_x, T *s_cosq, T *d_Tbody){
   load_Tb(s_x,s_Tb,d_Tbody,s_cosq,s_sinq,s_dTb);
   hd__syncthreads();
   // then compute Tbody -> T
   compute_T_TA_J(s_Tb,s_T);
   hd__syncthreads();
   // then computde T, dTbody -> dT
   T *s_dTp = &s_dTb[16*NUM_POS]; // 16*NUM_POS so 32*NUM_POS b/c using compressed dTb
   compute_dT_dTA_dJ(s_Tb,s_dTb,s_T,s_dT,s_dTp);
   hd__syncthreads();
   //compute the hand position and derivative use sinq as temp space
   compute_eePos(s_T,s_eePos,s_dT,s_deePos,s_sinq);
}
template <typename T>
__host__ __forceinline__
void compute_eePos_scratch(T *x, T *eePos){
   T s_cosq[NUM_POS];         T s_sinq[NUM_POS];      T s_Tb[36*NUM_POS];  
   T s_T[36*NUM_POS];         T Tbody[36*NUM_POS];
   initT<T>(Tbody);           load_Tb(x,s_Tb,Tbody,s_cosq,s_sinq);   
   compute_T_TA_J(s_Tb,s_T);  compute_eePos(s_T,eePos);
}
template <typename T>
__host__ __device__ __forceinline__
void compute_eeVel(T *s_T, T *s_eePos, T *s_TbTdt, T *s_eePosdt, T *s_T_dx = nullptr, T *s_T_dt_dx = nullptr, T *s_eePosVeldx = nullptr, int ld_grad = 0){
   int start, delta; singleLoopVals(&start,&delta);
   #ifdef __CUDA_ARCH__
      bool thread0Flag = threadIdx.x == 0 && threadIdx.y == 0;
   #else
      bool thread0Flag = 1;
   #endif
   bool derivFlag = s_T_dx != nullptr && s_T_dt_dx != nullptr && s_eePosVeldx != nullptr;
   T *Tee = &s_T[(NUM_POS-1)*36];
   T *Tee_dt = &s_TbTdt[(2*NUM_POS-1)*16]; //First 16*NUM_POS is Tbdt then next 16*NUM_POS is Tdt
   // compute initial helper terms
   T factor3 = Tee[6]*Tee[6] + Tee[10]*Tee[10];
   T factor4 = Tee[2]*Tee[2] + factor3;
   T factor5 = Tee[1]*Tee[1] + Tee[0]*Tee[0];
   T sqrtf3 = sqrt(factor3);
   T invf3 = static_cast<T>(1)/factor3;
   T invf4 = static_cast<T>(1)/factor4;
   T invf5 = static_cast<T>(1)/factor5;
   // T dsqrtTerm = (Tee[6]*Tee_dt[10] + Tee[10]*Tee_dt[6])/sqrtf3;
   if (thread0Flag){
      // then compute ee_pos
      s_eePos[0] = Tee[0]*static_cast<T>(EE_ON_LINK_X) + Tee[4]*static_cast<T>(EE_ON_LINK_Y) + Tee[8]*static_cast<T>(EE_ON_LINK_Z)  + Tee[12];
      s_eePos[1] = Tee[1]*static_cast<T>(EE_ON_LINK_X) + Tee[5]*static_cast<T>(EE_ON_LINK_Y) + Tee[9]*static_cast<T>(EE_ON_LINK_Z)  + Tee[13];
      s_eePos[2] = Tee[2]*static_cast<T>(EE_ON_LINK_X) + Tee[6]*static_cast<T>(EE_ON_LINK_Y) + Tee[10]*static_cast<T>(EE_ON_LINK_Z) + Tee[14];
      s_eePos[3] = (T) atan2(Tee[6],Tee[10]);
      s_eePos[4] = (T) atan2(-Tee[2],sqrtf3);
      s_eePos[5] = (T) atan2(Tee[1],Tee[0]);
      // and ee_vel
      s_eePosdt[0] = Tee_dt[0]*static_cast<T>(EE_ON_LINK_X) + Tee_dt[4]*static_cast<T>(EE_ON_LINK_Y) + Tee_dt[8]*static_cast<T>(EE_ON_LINK_Z)  + Tee_dt[12];
      s_eePosdt[1] = Tee_dt[1]*static_cast<T>(EE_ON_LINK_X) + Tee_dt[5]*static_cast<T>(EE_ON_LINK_Y) + Tee_dt[9]*static_cast<T>(EE_ON_LINK_Z)  + Tee_dt[13];
      s_eePosdt[2] = Tee_dt[2]*static_cast<T>(EE_ON_LINK_X) + Tee_dt[6]*static_cast<T>(EE_ON_LINK_Y) + Tee_dt[10]*static_cast<T>(EE_ON_LINK_Z) + Tee_dt[14];
      // TODO: something is wrong lets zero out the rpy stuff for now
      s_eePosdt[3] = 0;//-Tee[6]*Tee_dt[10] + Tee[10]*Tee_dt[6]; // don't divide by f3 for now as we will use for temp memeory in deriv comp
      s_eePosdt[4] = 0;// Tee[2]*dsqrtTerm  - sqrtf3*Tee_dt[2];  // don't divide by f4 for now as we will use for temp memeory in deriv comp
      s_eePosdt[5] = 0;//-Tee[1]*Tee_dt[0]  + Tee[0]*Tee_dt[1];  // don't divide by f5 for now as we will use for temp memeory in deriv comp
   }
   hd__syncthreads();
   if (derivFlag){
      // then compute all dk in parallel (note dqd is 0 so only dq needed)
      #pragma unroll
      for (int k = start; k < STATE_SIZE_PDDP; k += delta){
         T *T_dtdx = &s_T_dt_dx[16*k];
         if (k < NUM_POS){
            T *T_dx = &s_T_dx[16*k];
            // first the positional derivs for xyz
            s_eePosVeldx[k*ld_grad]      = T_dx[0]*static_cast<T>(EE_ON_LINK_X) + T_dx[4]*static_cast<T>(EE_ON_LINK_Y) + T_dx[8]*static_cast<T>(EE_ON_LINK_Z)  + T_dx[12];
            s_eePosVeldx[k*ld_grad + 1]  = T_dx[1]*static_cast<T>(EE_ON_LINK_X) + T_dx[5]*static_cast<T>(EE_ON_LINK_Y) + T_dx[9]*static_cast<T>(EE_ON_LINK_Z)  + T_dx[13];
            s_eePosVeldx[k*ld_grad + 2]  = T_dx[2]*static_cast<T>(EE_ON_LINK_X) + T_dx[6]*static_cast<T>(EE_ON_LINK_Y) + T_dx[10]*static_cast<T>(EE_ON_LINK_Z) + T_dx[14];
            // then the positional derivs for rpy
            s_eePosVeldx[k*ld_grad + 3]  = 0;//(-Tee[6]*T_dx[10] + Tee[10]*T_dx[6])*invf3;
            // T sqrtf3_dx = (Tee[6]*T_dx[6] + Tee[10]*T_dx[10])/sqrtf3;
            s_eePosVeldx[k*ld_grad + 4]  = 0;//(Tee[2]*sqrtf3_dx -  sqrtf3*T_dx[2])*invf4;
            s_eePosVeldx[k*ld_grad + 5]  = 0;//(-Tee[1]*T_dx[0]  +  Tee[0]*T_dx[1])*invf5;
            // then the velcity derivs for xyz
            s_eePosVeldx[k*ld_grad + 6]  = T_dtdx[0]*static_cast<T>(EE_ON_LINK_X) + T_dtdx[4]*static_cast<T>(EE_ON_LINK_Y) + T_dtdx[8]*static_cast<T>(EE_ON_LINK_Z)  + T_dtdx[12];
            s_eePosVeldx[k*ld_grad + 7]  = T_dtdx[1]*static_cast<T>(EE_ON_LINK_X) + T_dtdx[5]*static_cast<T>(EE_ON_LINK_Y) + T_dtdx[9]*static_cast<T>(EE_ON_LINK_Z)  + T_dtdx[13];
            s_eePosVeldx[k*ld_grad + 8]  = T_dtdx[2]*static_cast<T>(EE_ON_LINK_X) + T_dtdx[6]*static_cast<T>(EE_ON_LINK_Y) + T_dtdx[10]*static_cast<T>(EE_ON_LINK_Z) + T_dtdx[14];
            // and final velocity for rpy
            // T r_1 = (-T_dx[6]*Tee_dt[10] - Tee[6]*T_dtdx[10] + T_dx[10]*Tee_dt[6] + Tee[10]*T_dtdx[6])*invf3;
            // T r_2 = s_eePosdt[3]*2*(Tee[6]*T_dx[6] + Tee[10]*T_dx[10])*invf3*invf3;
            // T dsqrtTerm_dx = (T_dx[6]*Tee_dt[10] + Tee[6]*T_dtdx[10] + T_dx[10]*Tee_dt[6] + Tee[10]*T_dtdx[6])/sqrtf3 + 
            //                  (Tee[6]*Tee_dt[10] + Tee[10]*Tee_dt[6])*sqrtf3_dx;
            // T p_1 = (T_dx[2]*dsqrtTerm + Tee[2]*dsqrtTerm_dx - sqrtf3_dx*Tee_dt[2] - sqrtf3*T_dtdx[2])*invf4;
            // T p_2 = s_eePosdt[4]*2*(Tee[2]*T_dx[2] + Tee[6]*T_dx[6] + Tee[10]*T_dx[10])*invf4*invf4;
            // T y_1 = (-T_dx[1]*Tee_dt[0] - Tee[1]*T_dtdx[0] + T_dx[0]*Tee_dt[1] + Tee[0]*T_dtdx[1])*invf5;
            // T y_2 = s_eePosdt[5]*2*(Tee[1]*T_dx[1] + Tee[0]*T_dx[0])*invf5*invf5;
            s_eePosVeldx[k*ld_grad + 9]  = 0;//r_1 - r_2;
            s_eePosVeldx[k*ld_grad + 10] = 0;//p_1 - p_2;
            s_eePosVeldx[k*ld_grad + 11] = 0;//y_1 - y_2;
         }
         // for the pos gradient is zero wrt qd and T_dx is zero wrt qd as well
         else{
            s_eePosVeldx[k*ld_grad]      = 0;
            s_eePosVeldx[k*ld_grad + 1]  = 0;
            s_eePosVeldx[k*ld_grad + 2]  = 0;
            s_eePosVeldx[k*ld_grad + 3]  = 0;
            s_eePosVeldx[k*ld_grad + 4]  = 0;
            s_eePosVeldx[k*ld_grad + 5]  = 0;
            s_eePosVeldx[k*ld_grad + 6]  = T_dtdx[0]*static_cast<T>(EE_ON_LINK_X) + T_dtdx[4]*static_cast<T>(EE_ON_LINK_Y) + T_dtdx[8]*static_cast<T>(EE_ON_LINK_Z)  + T_dtdx[12];
            s_eePosVeldx[k*ld_grad + 7]  = T_dtdx[1]*static_cast<T>(EE_ON_LINK_X) + T_dtdx[5]*static_cast<T>(EE_ON_LINK_Y) + T_dtdx[9]*static_cast<T>(EE_ON_LINK_Z)  + T_dtdx[13];
            s_eePosVeldx[k*ld_grad + 8]  = T_dtdx[2]*static_cast<T>(EE_ON_LINK_X) + T_dtdx[6]*static_cast<T>(EE_ON_LINK_Y) + T_dtdx[10]*static_cast<T>(EE_ON_LINK_Z) + T_dtdx[14];
            s_eePosVeldx[k*ld_grad + 9]  = 0;//(-Tee[6]*T_dtdx[10] + Tee[10]*T_dtdx[6])*invf3;
            // T dsqrtTerm_dx = (Tee[6]*T_dtdx[10] + Tee[10]*T_dtdx[6])/sqrtf3;
            s_eePosVeldx[k*ld_grad + 10] = 0;//(Tee[2]*dsqrtTerm_dx - sqrtf3*T_dtdx[2])*invf4;
            s_eePosVeldx[k*ld_grad + 11] = 0;//(-Tee[1]*T_dtdx[0] + Tee[0]*T_dtdx[1])*invf5;
         }
      }
   }
   hd__syncthreads();
   // then finish off the vels now that we aren't using for temp memory
   if (thread0Flag){
      s_eePosdt[3] = s_eePosdt[3]*invf3;
      s_eePosdt[4] = s_eePosdt[4]*invf4;
      s_eePosdt[5] = s_eePosdt[5]*invf5;
   }
}
template <typename T>
__host__ __forceinline__
void compute_eeVel_scratch(T *x, T *eePos, T *eeVel){
   T s_cosq[NUM_POS];         T s_sinq[NUM_POS];      T s_Tb[36*NUM_POS];  
   T s_T[36*NUM_POS];         T Tbody[36*NUM_POS];    T s_TbTdt[36*NUM_POS];
   initT<T>(Tbody);           load_Tbdt(x,s_Tb,Tbody,s_cosq,s_sinq,s_TbTdt);
   compute_T_TA_J<T>(s_Tb,s_T,nullptr,nullptr,s_TbTdt);
   compute_eeVel<T>(s_T,eePos,s_TbTdt,eeVel);
}
// need to pass in s_x and then temp memory for the rest
// s_Tb is 36*NUM_POS (only stored in first 16 of each)
// s_dTb_dx is 16*NUM_POS (2nd half is free) and use for s_T_dx_prev (16*NUM_POS)
// s_dTbTdt is 16*NUM_POS + 16*NUM_POS
// s_T is 36*NUM_POS (only stored in first 16 of each)
// s_T_dx is 16*NUM_POS and s_Tb_dt_dx is 16*NUM_POS -> temp1
// s_T_dt_dx is 16*2*NUM_POS and s_T_dt_dx prev is 16*2*NUM_POS -> temp 2, temp 3 b/c each 32*NUM_POS
// s_eePosVel_dx is 12*2*NUM_POS b/c q and qd
template <typename T>
__host__ __device__ __forceinline__
void compute_eePosVel_dx(T *s_x, T *s_Tb, T *d_Tb, T *s_cosq, T *s_sinq, T *s_Tb_dx, T *s_TbTdt, T *s_Tb_dt_dx, T *s_T, 
                         T *s_T_dx, T *s_T_dt_dx, T *s_T_dt_dx_prev, T *s_eePos, T *s_eeVel, T *s_eePosVel_dx, int ld_grad){
    // load in Tb, Tb_dx, Tb_dt
    load_Tbdtdx<T>(s_x,s_Tb,d_Tb,s_sinq,s_cosq,s_Tb_dx,s_TbTdt,s_Tb_dt_dx);
    hd__syncthreads();
    // then compute Tbody -> T & T_dt
    compute_T_TA_J<T>(s_Tb,s_T,nullptr,nullptr,s_TbTdt);
    T *s_Tb_dt = s_TbTdt;   T *s_T_dt = &s_TbTdt[16*NUM_POS];
    hd__syncthreads();
    // then computde T, dTbody -> dT
    T *s_T_dx_prev = &s_Tb_dx[16*NUM_POS]; // use 2nd half for space savings
    compute_T_dtdx<T>(s_Tb,s_Tb_dx,s_Tb_dt,s_Tb_dt_dx,s_T,s_T_dx,s_T_dt,s_T_dt_dx,s_T_dx_prev,s_T_dt_dx_prev);
    hd__syncthreads();
    //compute the hand position and velocity and its derivatives
    compute_eeVel<T>(s_T,s_eePos,s_TbTdt,s_eeVel,s_T_dx,s_T_dt_dx,s_eePosVel_dx,ld_grad);
}

template <typename T>
__host__ __device__ __forceinline__
void dynamics(T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody, T *s_eePos = nullptr, int reps = 1, T *s_eeVel = nullptr){
   #ifdef __CUDA_ARCH__
      __shared__ T s_I[36*NUM_POS]; // standard inertias -->  world inertias
      __shared__ T s_Icrbs[36*NUM_POS]; // Icrbs inertias
      __shared__ T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      __shared__ T s_temp[36*NUM_POS]; // temp work space (load Tbody mats into here) --> and JdotV
      __shared__ T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> and crm(twist) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      __shared__ T s_TA[36*NUM_POS]; // adjoint transpose --> crf(twist)
      __shared__ T s_W[6*NUM_POS]; // to store net wrenches  
      __shared__ T s_F[6*NUM_POS]; // to store forces in joint axis
      __shared__ T s_JdotV[6*NUM_POS]; // JdotV vectors
   #else
      T s_I[36*NUM_POS]; // standard inertias -->  world inertias
      T s_Icrbs[36*NUM_POS]; // Icrbs inertias
      T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      T s_temp[36*NUM_POS]; // temp work space (load Tbody mats into here) --> and JdotV
      T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> and crm(twist) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      T s_TA[36*NUM_POS]; // adjoint transpose --> crf(twist)
      T s_W[6*NUM_POS]; // to store net wrenches  
      T s_F[6*NUM_POS]; // to store forces in joint axis
      T s_JdotV[6*NUM_POS]; // JdotV vectors
   #endif
   for(int iter = 0; iter < reps; iter++){
      T *s_xk = &s_x[STATE_SIZE_PDDP*iter];
      T *s_uk = &s_u[NUM_POS*iter];
      T *s_qddk = &s_qdd[NUM_POS*iter];
      // load in I and Tbody (use W and F as temp mem)
      // use I_crbs for dTb if need to compute s_eeVel
      load_I(s_I,d_I);
      if (s_eeVel != nullptr){load_Tbdt(s_xk,s_temp,d_Tbody,s_F,s_W,s_Icrbs);}
      else{load_Tb(s_xk,s_temp,d_Tbody,s_F,s_W);}
      hd__syncthreads();
      // then compute Tbody -> T -> TA & J (T and Tbody in scratch mem)
      // use I_crbs for dTb and dT/dt if need to compute s_eeVel
      if (s_eeVel != nullptr){
        compute_T_TA_J(s_temp,s_temp2,s_TA,s_J,s_Icrbs); hd__syncthreads();
        compute_eeVel(s_temp2,s_eePos,s_Icrbs,s_eeVel); hd__syncthreads();
      }
      else{
        compute_T_TA_J(s_temp,s_temp2,s_TA,s_J); hd__syncthreads();
        // if we are asked for just eePos then compute that
        if (s_eePos != nullptr){compute_eePos(s_temp2,s_eePos); hd__syncthreads();}
      }
      // then compute Iworld, Icrbs, twists (in W) and clear temp and TA for later
      compute_Iw_Icrbs_twist(s_I,s_Icrbs,s_W,s_TA,s_J,s_xk,s_temp);
      hd__syncthreads();
      // then JdotV (twists in W)
      compute_JdotV(s_JdotV,s_W,s_J,s_xk,s_temp);
      hd__syncthreads();
      // finally compute F > biasForce(Tau) & [massMatrix|I] from twists in W and JdotV and Icrbs etc.
      T *s_M = s_temp2; // reuse scratch mem and note that s_TA cleared in inertia comp so use as scratch mem as well
      T *s_Tau = &s_temp2[2*NUM_POS*NUM_POS];
      compute_M_Tau(s_M, s_Tau, s_W, s_JdotV, s_F, s_Icrbs, s_W, s_J, s_I, s_xk, s_uk, s_temp, s_temp2, s_TA);
      hd__syncthreads();
      // invert Mass matrix -- assumes more threads than NUM_POS +1 by NUM_POS -- writes out [I|M^{-1}]
      #ifdef __CUDA_ARCH__
         int err = invertMatrix<T,NUM_POS,1>(s_M,s_F);
      #else
         int err = invertMatrix<T,NUM_POS>(s_M,s_F);
      #endif
      // TBD: DO SOMETHING WITH THE ERROR
      T *s_Minv = &s_temp2[NUM_POS*NUM_POS];
      hd__syncthreads();
      // finally compute qdd 
      compute_qdd(s_qddk,s_Minv,s_Tau);
   }
}

template <typename T>
__host__ __device__ __forceinline__
void dynamicsGradient(T *s_dqdd, T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody){
   #ifdef __CUDA_ARCH__
      __shared__ T s_I[36*NUM_POS]; // standard inertis -->  world inertias
      __shared__ T s_Icrbs[36*NUM_POS]; // crbs inertias (load dTbody here to start)
      __shared__ T s_TA[42*NUM_POS]; // adjoint transpose --> crf(twist)
      __shared__ T s_dTA[36*NUM_POS*NUM_POS]; // derive adjoint transpose --> dM --> dTwist, dJdotV, dW
      __shared__ T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      __shared__ T s_dJ[6*NUM_POS*NUM_POS]; // derivative of J
      __shared__ T s_JdotV[6*NUM_POS]; // JdotV vectors
      __shared__ T s_twist[6*NUM_POS]; // twist vectors
      __shared__ T s_W[6*NUM_POS]; // to store net wrenches 
      __shared__ T s_F[6*NUM_POS]; // to store forces in joint axis
      __shared__ T s_temp[36*NUM_POS]; // temp work space (load Tbody into here)
      __shared__ T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      __shared__ T s_temp3[36*NUM_POS*NUM_POS]; // compute dT here --> then compute dIw here
   #else
      T s_I[36*NUM_POS]; // standard inertis -->  world inertias
      T s_Icrbs[36*NUM_POS]; // crbs inertias (load dTbody here to start)
      T s_TA[42*NUM_POS]; // adjoint transpose --> crf(twist)
      T s_dTA[36*NUM_POS*NUM_POS]; // derive adjoint transpose --> dM --> dTwist, dJdotV, dW
      T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      T s_dJ[6*NUM_POS*NUM_POS]; // derivative of J
      T s_JdotV[6*NUM_POS]; // JdotV vectors
      T s_twist[6*NUM_POS]; // twist vectors
      T s_W[6*NUM_POS]; // to store net wrenches 
      T s_F[6*NUM_POS]; // to store forces in joint axis
      T s_temp[36*NUM_POS]; // temp work space (load Tbody into here)
      T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      T s_temp3[36*NUM_POS*NUM_POS]; // compute dT here --> then compute dIw here
   #endif
   // compute Tbody and dTbody (in temp and Icrbs) and use W and F as temp memory
   T *s_Tb = s_temp; // 36*NUM_POS
   T *s_dTb = s_temp2; // 16*NUM_POS
   load_Tb(s_x,s_Tb,d_Tbody,s_W,s_F,s_dTb);
   load_I(s_I,d_I);
   hd__syncthreads();
   T *s_T = s_Icrbs; // 16*NUM_POS
   // then compute Tbody -> T -> TA & J (T and Tbody in scratch mem)
   compute_T_TA_J(s_Tb,s_T,s_TA,s_J);
   hd__syncthreads();
   // then computde dTbody,T, TA -> dT -> dTA & dJ
   T *s_dT = s_temp3; // 36*NUM_POS
   T *s_dTp = &s_temp2[16*NUM_POS]; // 16*NUM_POS so 32*NUM_POS
   compute_dT_dTA_dJ(s_Tb,s_dTb,s_T,s_dT,s_dTp,s_TA,s_dTA,s_dJ);
   hd__syncthreads();
   // compute Iworld, Icrbs, twists, and dIw (in dTA b/c now done with that) and clear temp and TA for later
   compute_Iw_Icrbs_twist(s_I,s_Icrbs,s_twist,s_TA,s_J,s_x,s_temp,s_dTA,s_temp2);
   T *s_dIw = s_dTA;
   hd__syncthreads();
   // now finish normal comp before doing rest of dervatives so compute JdotV
   compute_JdotV(s_JdotV,s_twist,s_J,s_x,s_temp);
   hd__syncthreads();
   // then compute F > biasForce(Tau) & [massMatrix|I]
   T *s_M = s_temp2; // reuse scratch mem and note that s_TA cleared in inertia comp so use as scratch mem as well
   T *s_Tau = &s_temp2[2*NUM_POS*NUM_POS];
   compute_M_Tau(s_M, s_Tau, s_W, s_JdotV, s_F, s_Icrbs, s_twist, s_J, s_I, s_x, s_u, s_temp, s_temp2, s_TA);
   hd__syncthreads();
   // invert Mass matrix -- assumes more threads than NUM_POS +1 by NUM_POS -- writes out [I|M^{-1}]
   #ifdef __CUDA_ARCH__
      int err = invertMatrix<T,NUM_POS,1>(s_M,s_F);
   #else
      int err = invertMatrix<T,NUM_POS>(s_M,s_F);
   #endif
   // TBD: DO SOMETHING WITH THE ERROR
   T *s_Minv = &s_temp2[NUM_POS*NUM_POS];
   hd__syncthreads();
   // finally compute qdd
   compute_qdd(s_qdd,s_Minv,s_Tau);
   
   // ---------------------------------------------
   // note we now have:
   //    -J, dJ in s_J, s_dJ
   //    -Iw, dIw in s_I, s_dTA
   //    -Icrbs in s_Icrbs
   //    -invM in s_temp2
   //    -C    in s_temp2
   //    -twists in s_twist
   //    -netW   in s_W
   //    -JdotV in s_JdotV
   //    -u in s_u
   //    -qdd,qd in s_x
   // we should also be about maxed out in shared memory so we need
   // to loop to finish this up and note that we still have free:
   // s_temp[36*NB],s_F[6*NB],s_TA[42*NB = 6*NB*NB],s_temp3[36*NB*NB]
   // ---------------------------------------------

   // IF WE CAN LOOP TO REDUCE THE USAGE OF s_temp3 down to 36*NB we can reduce it in size b/c not using big before here

   // dM is going to be size NB*NB*NB so can store in s_temp3 as long as NB < 36
   T *s_dM = s_temp3;
   compute_dM(s_dM,s_Icrbs,s_dIw,s_J,s_dJ,s_F,s_TA);
   hd__syncthreads();
   // the comptue first half of dqdd with relation to dM (dqdd_du is just Minv which we already have for arm)
   compute_dqdd_dM(s_dqdd,s_dM,s_Minv,s_qdd,s_temp);
   hd__syncthreads();
   // now we need to form dTau to compute the other half of dqdd_dx
   // unfortunately we need to loop to save memory
   // note we can reuse the temp memory space as we are done with dM
   // T *s_dTwist = s_temp3;
   // T *s_dTwistp = &s_temp3[12*NUM_POS];
   // T *s_dJdotV = &s_temp3[2*12*NUM_POS];
   // T *s_dJdotVp = &s_temp3[3*12*NUM_POS];
   // T *s_dWb = &s_temp3[4*12*NUM_POS];
   // T *s_dWp = &s_temp3[5*12*NUM_POS];
   // T *s_dTau = &s_temp3[6*12*NUM_POS];
   // compute_dTau(s_dTau,s_W,s_dWb,s_dWp,s_JdotV,s_dJdotV,s_dJdotVp,s_twist,s_dTwist,s_dTwistp,s_I,s_dIw,s_J,s_dJ,s_x,s_temp,s_temp2,s_Icrbs,s_F);
   T *s_dTwist = s_temp3;
   T *s_dJdotV = &s_temp3[6*(2*NUM_POS)*NUM_POS];
   T *s_dWb = &s_temp3[6*(4*NUM_POS)*NUM_POS];
   // so compute dTwist then dJdotV then dWb
   compute_dtwist(s_dTwist,s_J,s_dJ,s_x);
   hd__syncthreads();
   compute_dJdotV(s_dJdotV,s_twist,s_dTwist,s_J,s_dJ,s_x,s_temp,s_TA);
   hd__syncthreads();
   compute_dWb(s_dWb,s_JdotV,s_dJdotV,s_twist,s_dTwist,s_I,s_dIw,s_temp,s_TA,s_F);
   hd__syncthreads();
   // use those to comptue dTau
   T *s_dTau = s_temp;
   compute_dTau(s_dTau,s_dWb,s_W,s_J,s_dJ);
   hd__syncthreads();
   // finally dqdd += invH*dTau to compute the second part of dqdd_dx and load dqdd_du
   finish_dqdd(s_dqdd,s_dTau,s_Minv);
}
/*** KINEMATICS AND DYNAMICS HELPERS ***/
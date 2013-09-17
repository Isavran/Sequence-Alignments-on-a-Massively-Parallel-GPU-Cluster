/**
 * g_interface.h
 *
 *  Created on: Jan 18, 2012
 *      Author: ygao
 *  
 *  Updated by: Ibrahim Savran
 *    Bigger Group-Size
 *    Unncessary memory allocations removed
 * 
 */

#ifndef G_INTERFACE_H_
#define G_INTERFACE_H_

#include <cuda.h>
#include "SeqDist.h"
extern FILE *glb_outFiles[10];

//avoid GPU confict with other user of the workstation
//#define DEBUG_TACHYON 		0

////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////
/*CPU and GPU switch*/
#define GPU_ACC_ON 			1

/*texture cache usage*/
//#define TEXTURE_USAGE	 	0
#define SHARE_USAGE 		0

/* multi gpu switch */
#define MULTI_GPU 			0
#define GPU_NUM 			1
/* GPU arch */
//#define CUDA_GT200 			0
#define CUDA_FERMI 		0

#ifdef CUDA_FERMI
#undef TEXTURE_USAGE
#undef SHARE_USAGE
#endif

#define GROUP_SIZE  		256///savran	32

#ifndef GPU_ACC_ON
#define CONCURRENT_GROUPS 	1
#else
// TODO: adjust this automatically
//GPU feature
#define MAX_SM 				1///savran		30
//choose by hand, according to the register usage
//and the how many thread per block(warps per block)
//>>>>>>>>>>>>>>>
#define WARPS_PER_BLOCK 	1
//calculated from occupancy calculator
#define WARPS_PER_SM 		1

#define IDEAL_WARPS_PER_CARD (MAX_SM*WARPS_PER_SM)

//choose by hand to minimize the distance
//between IDEAL_WARPS_PER_CARD and REAL_WARPS_PER_CARD
//>>>>>>>>>>>>>>>
#define CONCURRENT_GROUPS 	1 ///savran 12
#define REAL_WARPS_PER_CARD (GROUP_SIZE*CONCURRENT_GROUPS)
#endif
//#define FMAT_LINES 		3
//packet element for MPI send
typedef struct s_mpi_dist{
	int i;
	int j;
	FLOAT_T dDist;
} t_mpi_dist;

typedef struct s_Data_GPU{
	CUdevice 	cuDevice;
	CUcontext 	cuContext;
	CUmodule 	cuModule;
	CUfunction 	cuFunction;

	CUdeviceptr cu_acSequences;
	CUdeviceptr cu_acSequences_ro;
	CUdeviceptr cu_anLen;

	/*for move matrix in GPU*/
//	CUdeviceptr cu_aaMoves[GROUP_SIZE * CONCURRENT_GROUPS];
//	CUdeviceptr cu_aanHA[GROUP_SIZE*CONCURRENT_GROUPS];
//	CUdeviceptr cu_aanHB[GROUP_SIZE*CONCURRENT_GROUPS];
//	CUdeviceptr cu_aadFMatrix[GROUP_SIZE * CONCURRENT_GROUPS];
	//pass this to device
//CUdeviceptr	cu_aaMoves_p;
//	CUdeviceptr cu_aanHA_p;
//	CUdeviceptr cu_aanHB_p;
//	CUdeviceptr cu_aadFMatrix_p;

	//pass this to device
	CUdeviceptr cu_dDist_p;		
//	FLOAT_T 	*dDist[1000];	///bunlar dinamik olacak///OPT
//	CUdeviceptr cu_dDist[1000];

	FLOAT_T 	*dDistBig;		/// gnum*grp*grp;
	FLOAT_T 	*cu_dDistBig;	/// gnum*grp*grp;

	CUdeviceptr cu_adLookup;
	int 		cu_maxCongrp;
#ifdef TEXTURE_USAGE
	//texture related
	CUtexref 	cu_texLookUp;
	CUtexref 	cu_texacA;
	CUtexref 	cu_texacB;
#endif
} t_Data_GPU;

typedef struct s_cu_Thread{
	int 		numtasks;
	int 		rank;
	int 		threadNum;	///GPU_NUM =1
	int 		threadId;
	int 		gNum;
	t_mpi_dist 	*distElem;
	int 		distElemI;
	///data for CPU
	t_Data 		*tData;
	///data for GPU
	t_Data_GPU 	*tData_GPU;
	///adlookup
	FLOAT_T		*adLookUp;
} t_cu_Thread;

#ifdef __cplusplus
extern "C" {
#endif
CUresult initCUDA(t_cu_Thread *cu_Thread);
int pack_dist(t_mpi_dist* distElem, int elemNum, FLOAT_T *dDist, int nDistSize,
		int init, int orgDistSize);
int resizeData(t_Data *cpuData, int newN);
/** rearrange the seq to meet memory corlace needs*/
int seqRearrange(t_Data *cpuData, char* acSequences);
/** kernel launch */
int cuNeedlemanWunsch(int gi, int gj, t_cu_Thread *cu_Thread,
		t_mpi_dist* distElem, int distElemI, int con_grp);
/** allocate GPU resource */
int cuResAlloc(t_cu_Thread *cu_Thread, FILE *fhandle);
/**timer*/
void start_time();
double get_time();
#ifdef __cplusplus
}
#endif
#endif /** G_INTERFACE_H_ */

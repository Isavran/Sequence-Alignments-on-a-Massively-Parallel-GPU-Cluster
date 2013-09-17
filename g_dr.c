/**
 * g_interface.cu
 *
 *  Created on: Jan 18, 2012
 *      Author: ygao
 *   
 *  Updated by: Ibrahim Savran
 *    Score values are written to the files
 *    Blocks of threads are called in total
 *    (no Concurrent group calls needed) 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>
#include "g_interface.h"

static struct timeval tstart, tstartlocal;
/// Timing
void start_time(){
	gettimeofday(&tstart, NULL);
}

double get_time(){
	struct timeval tfinish;
	long sec,usec;

	gettimeofday(&tfinish, NULL);
	sec = tfinish.tv_sec - tstart.tv_sec;
	usec = tfinish.tv_usec - tstart.tv_usec;
	return (double)(sec + 1e-6*usec);
}
/// end of timing

void startlocal_time(){
	gettimeofday(&tstartlocal, NULL);
}

double getlocal_time(){
	struct timeval tfinishlocal;
	long sec, usec;

	gettimeofday(&tfinishlocal, NULL);
	sec = tfinishlocal.tv_sec - tstartlocal.tv_sec;
	usec = tfinishlocal.tv_usec - tstartlocal.tv_usec;
	return (double)(sec + 1e-6*usec);
}

CUresult initCUDA(t_cu_Thread *cu_Thread){

	CUresult status = CUDA_SUCCESS;
	t_Data_GPU *tData_GPU = cu_Thread->tData_GPU;
	int rank = cu_Thread->rank;
	int gNum = cu_Thread->gNum;
	int threadId = cu_Thread->threadId;
	int threadNum = cu_Thread->threadNum;
	int deviceNum;
	int deviceId = 0;
	
	status = cuInit(0);
	if (CUDA_SUCCESS != status){fprintf(stderr, "cuInit failed:%10d status %10d\n", __LINE__,status);	goto Error; exit(0);}

	status = cuDeviceGet(&(tData_GPU->cuDevice), deviceId);
	if (CUDA_SUCCESS != status){	fprintf(stderr, "cuDeviceGet failed:%d\n", __LINE__);	goto Error; exit(0); }

	status = cuCtxCreate(&(tData_GPU->cuContext), 0, tData_GPU->cuDevice);
	if (CUDA_SUCCESS != status ){	fprintf(stderr, "cuCtxCreate failed:%d\n", __LINE__);	goto Error; exit(0);}
	status = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
	if (CUDA_SUCCESS != status){
		fprintf(stderr, "cuCtx bigger cache  failed:%d\n", __LINE__);
		goto Error; exit(0);
	}


	status = cuModuleLoad(&(tData_GPU->cuModule), "g_kernel.cubin");
	if (CUDA_SUCCESS != status) {
		fprintf(stderr, "cuModuleLoad failed:%d status %5d \n", __LINE__, status /*, cudaGetErrorString(status)*/);
		goto Error; exit(0);
	}

	status = cuModuleGetFunction(&(tData_GPU->cuFunction), tData_GPU->cuModule, "needlemanWunsch_CU");
	if (CUDA_SUCCESS != status) {
		fprintf(stderr, "cuModuleGetFunction failed:%d:%d\n", status, __LINE__);
		goto Error; exit(0);
	}

	return CUDA_SUCCESS;

	Error: cuCtxDetach(tData_GPU->cuContext);
	return status;
}

int partition( int *allIndex, FLOAT_T a[], int l, int r) {
	int  i, j,tempi;
	FLOAT_T pivot, t;
	pivot = a[l];
	i = l;
	j = r+1;
	while(1){
		do ++i; while( a[i] <= pivot && i <= r );
		do --j; while( a[j] > pivot );
		if( i >= j ) break;
		t = a[i];	a[i] = a[j]; a[j] = t;
		tempi = allIndex[i];	allIndex[i]=allIndex[j];	allIndex[j] = tempi;
	}
	t = a[l];	a[l] = a[j];	a[j] = t;
	tempi = allIndex[l];	allIndex[l]=allIndex[j];	allIndex[j] = tempi;
	return j;
}///end partition

void quickSort( int *allIndex, FLOAT_T a[], int l, int r){
   int j;
   if( l < r ) {
   	// divide and conquer
        j = partition( allIndex, a, l, r);
       quickSort(allIndex, a, l, j-1);
       quickSort(allIndex, a, j+1, r);
   }
}///end quickSort


int cuResAlloc(t_cu_Thread *cu_Thread, FILE *fhandle) {
	int i, j, k, l;

	CUresult status = CUDA_SUCCESS;
	t_Data_GPU *tData_GPU = cu_Thread->tData_GPU;
	t_Data *tData = cu_Thread->tData;
	int nN = tData->nSeq;
	int nM = tData->nMaxLen;
	int rank = cu_Thread->rank;
	int gNum = cu_Thread->gNum; ///savran
	char* acSequences_ch = tData->acSequences;
	char* acSequences = (char*) malloc(nN * nM * sizeof(char));
	char* acSequences_ro = (char*) malloc(nN * nM * sizeof(char));
	int *anLen = tData->anLen;
	
	FLOAT_T* adLookUp = cu_Thread->adLookUp;
	FLOAT_T* Allmax = tData->Allmax; ///allindex yer acildi
	int *allIndex 	= tData->allIndex;

	for (int jj=0; jj < nN/ GROUP_SIZE +1; jj++)
			allIndex[jj] = jj; /// indexi siraladik

	//convert the character into number
	for(i = 0; i < nN; i++){
		for(j = 0; j < nM; j++){
			if(acSequences_ch[i*nM + j] == 'A')			acSequences[i*nM + j] = 0;
			else if(acSequences_ch[i*nM + j] == 'C')	acSequences[i*nM + j] = 1;
			else if(acSequences_ch[i*nM + j] == 'T')	acSequences[i*nM + j] = 2;
			else if(acSequences_ch[i*nM + j] == 'G')	acSequences[i*nM + j] = 3;
			else										acSequences[i*nM + j] = -1;
		}
	}

	/**/
	//FLOAT_T Allmax[nN / GROUP_SIZE+2];
	for (l = 0; l < nN / GROUP_SIZE-1; l++ ){
		//reorder the input to meet the coalesced access
		k = l * GROUP_SIZE * nM;
		///group max burada bulunur
		for (i = 0; i < nM; i++ ){	
			for (j = 0; j < GROUP_SIZE; j++){	///nm maxlen
				acSequences_ro[k] =		acSequences[l * GROUP_SIZE * nM + j * nM + i];
				k++;		
			}
		}///find cumulative...
	} 
	
	/* For load balance 
	  for (l = 0; l < nN / GROUP_SIZE-1; l++ ){
		//reorder the input to meet the coalesced access
		k = l * GROUP_SIZE;
		///group max burada bulunur
		int maxx=0; /// butun blogun maximumu
			for (j = 0; j < GROUP_SIZE; j++){	///nm maxlen	
				if (anLen[k+j] > maxx) maxx = anLen[k+j];
			}
		Allmax[l] = (FLOAT_T)maxx/400.0; /// normalized 
			///find cumulative...
	}*/
	/*
	FLOAT_T cum=0.0; /// cumulative
	for (l = 0; l < nN / GROUP_SIZE-CONCURRENT_GROUPS; l++ ){ 
		FLOAT_T val = cum * Allmax[l];
		cum +=Allmax[l];
		Allmax[l] = val; /// normalized 
			///find cumulative...
	}
	int numtasks = cu_Thread->numtasks;
	*/
	/*FLOAT_T wl1=0.0, wl2=0.0;
	for (l = 0; l < nN / GROUP_SIZE-CONCURRENT_GROUPS; l++ ){ 
		///find cumulative...
		if (l % numtasks == rank )	
			wl1+=Allmax[l];
	}
		fprintf(fhandle, "work load-1 %5.5e\n", wl1);

	quickSort( allIndex, Allmax, 0, nN / GROUP_SIZE-CONCURRENT_GROUPS); /// ilk ve son

/*	*//*
	for (l = 0; l < nN / GROUP_SIZE-CONCURRENT_GROUPS; l++ ){ 
		///find cumulative...
		//fprintf(fhandle, "wl2 %5.5e\n", wl2);
		if (allIndex[l] % numtasks == rank )	
			wl2+=Allmax[l];
	}

	fprintf(fhandle, "work load-2 %5.5e\n", wl2);
*/
	//Copy Sequence into GPU
	status = cuMemAlloc(&(tData_GPU->cu_acSequences), sizeof(char) * nM * nN);
	if (status != CUDA_SUCCESS){	fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);	goto error; }

	status = cuMemAlloc(&(tData_GPU->cu_acSequences_ro), sizeof(char) * nM * nN);
	if (status != CUDA_SUCCESS){	fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);	goto error;	}

	status = cuMemcpyHtoD(tData_GPU->cu_acSequences, acSequences, sizeof(char) * nM * nN);
	if (status != CUDA_SUCCESS){ fprintf(stderr, "cuMemcpyHtoD failed:%d\n", __LINE__);	goto error;	}

	status = cuMemcpyHtoD(tData_GPU->cu_acSequences_ro, acSequences_ro, sizeof(char) * nM * nN);
	if (status != CUDA_SUCCESS){	fprintf(stderr, "cuMemcpyHtoD failed:%d\n", __LINE__);	goto error;}

	//alloc and copy anLen into GPU
	//printf("anlen  %5d \n",sizeof(int) * nN);
	status = cuMemAlloc(&(tData_GPU->cu_anLen), sizeof(int) * nN);
	if (status != CUDA_SUCCESS){	fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);	goto error;}

	status = cuMemcpyHtoD(tData_GPU->cu_anLen, anLen, sizeof(int) * nN);
	if (status != CUDA_SUCCESS){ fprintf(stderr, "cuMemcpyHtoD failed:%d\n", __LINE__);	goto error;	}

	// alloc the pointers in GPU 
	// status = cuMemAlloc(&(tData_GPU->cu_dDist_p), CONCURRENT_GROUPS * sizeof(FLOAT_T*));
	status = cuMemAlloc(&(tData_GPU->cu_dDist_p), gNum * sizeof(FLOAT_T*)); ///OPT degisecek
	if (status != CUDA_SUCCESS){	fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);	goto error; }
/*
	status = cuMemAlloc(&(tData_GPU->cu_aaMoves_p),
			GROUP_SIZE * CONCURRENT_GROUPS * sizeof(char*));
	if (status != CUDA_SUCCESS){
		fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);
		goto error;
	}*/

/*	//alloc cu_aaMoves
	for (j = 0; j < (GROUP_SIZE * CONCURRENT_GROUPS); j++){
		status = cuMemAlloc(&(tData_GPU->cu_aaMoves[j]),
				GROUP_SIZE * (nM + 1) * (nM + 1) * sizeof(char));
		if (status != CUDA_SUCCESS){
			fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);
			goto error;
		}
	}*/
	//dDist in both GPU and CPU end
	//savran for (j = 0; j < CONCURRENT_GROUPS; j++){  ///OPT Degisecek
/*  not needed	
	for (j = 0; j < gNum; j++ ){
		status = cuMemAlloc(&(tData_GPU->cu_dDist[j]), GROUP_SIZE * GROUP_SIZE * sizeof(FLOAT_T));
		if (status != CUDA_SUCCESS){ fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);	goto error;	}

		status = cuMemAllocHost((void**) &(tData_GPU->dDist[j]), GROUP_SIZE * GROUP_SIZE * sizeof(FLOAT_T));
		if (status != CUDA_SUCCESS){ fprintf(stderr, "cuMemAllocHost failed:%d\n", __LINE__);	goto error;	}
	}
	/*notneeded*/
	/// sadece bigler olacak...
	///big
	status = cuMemAlloc(&(tData_GPU->cu_dDistBig), gNum * GROUP_SIZE * GROUP_SIZE * sizeof(FLOAT_T));
	if (status != CUDA_SUCCESS){ fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);	goto error;}

	status = cuMemAllocHost((void**) &(tData_GPU->dDistBig), gNum * GROUP_SIZE * GROUP_SIZE * sizeof(FLOAT_T));
	if (status != CUDA_SUCCESS){ fprintf(stderr, "cuMemAllocHost failed:%d\n", __LINE__);	goto error;}
	

/*	//copy the pointers to GPU
	status = cuMemcpyHtoD(tData_GPU->cu_aaMoves_p, tData_GPU->cu_aaMoves,
			sizeof(char*) * GROUP_SIZE * CONCURRENT_GROUPS);
	if (status != CUDA_SUCCESS){
		fprintf(stderr, "cuMemcpyHtoD failed:%d\n", __LINE__);
		goto error;
	}//*/

	//status = cuMemcpyHtoD(tData_GPU->cu_dDist_p, tData_GPU->cu_dDist,
	//		sizeof(FLOAT_T*) * CONCURRENT_GROUPS); ///OPT
	//status = cuMemcpyHtoD(tData_GPU->cu_dDist_p, tData_GPU->cu_dDist, sizeof(FLOAT_T*) * gNum); 
	//if ( status != CUDA_SUCCESS ){	fprintf(stderr, "cuMemcpyHtoD failed:%d\n", __LINE__); goto error;}

	//alloc and copy for the cu_adLookup
	status = cuMemAlloc(&(tData_GPU->cu_adLookup), N_BASES * N_BASES * sizeof(FLOAT_T));
	if (status != CUDA_SUCCESS){fprintf(stderr, "cuMemMalloc failed:%d\n", __LINE__);	goto error;	}

	status = cuMemcpyHtoD(tData_GPU->cu_adLookup, adLookUp, N_BASES * N_BASES * sizeof(FLOAT_T));
	if (status != CUDA_SUCCESS){	fprintf(stderr, "cuMemcpyHtoD failed:%d\n", __LINE__);	goto error;}
	return CUDA_SUCCESS;	///
	error: return status;	
}

int cuNeedlemanWunsch(int gi, int gj, t_cu_Thread *cu_Thread,
					t_mpi_dist* distElem, int distElemI, int con_grp){
	CUresult status = CUDA_SUCCESS;
	t_Data_GPU *tData_GPU = cu_Thread->tData_GPU;
	t_Data *tData = cu_Thread->tData;
	CUfunction cuFunction = tData_GPU->cuFunction;
	int nM = tData->nMaxLen;
	int locDistElemI = distElemI;
	int threadNum = cu_Thread->threadNum;
	int i, j, k;
	int rank = cu_Thread->rank;

	FILE *fhandle = glb_outFiles[cu_Thread->threadId];

	void *cuPara[] = { 		
			&tData_GPU->cu_adLookup,
			&tData_GPU->cu_acSequences,
			&tData_GPU->cu_acSequences_ro,
			&gi,
			&gj,
			&tData_GPU->cu_anLen,
//			&tData_GPU->cu_dDist_p,
			&tData_GPU->cu_dDistBig,	///yeni big
			//&tData_GPU->cu_aaMoves_p,
//			&tData_GPU->cu_aanHA_p,
//			&tData_GPU->cu_aanHB_p,
			&nM, &threadNum};
		//	cudaPrintfInit();

	//if(con_grp/WARPS_PER_BLOCK > 0){	///warppersblock=1
//	if(con_grp/1 == 0){	///warppersblock=1  
		//int group = con_grp%WARPS_PER_BLOCK==0 ?(con_grp/WARPS_PER_BLOCK):((con_grp/WARPS_PER_BLOCK)+1);
		int group = con_grp;//%WARPS_PER_BLOCK==0 ?(con_grp/WARPS_PER_BLOCK):((con_grp/WARPS_PER_BLOCK)+1);

		status = cuLaunchKernel( cuFunction, 	GROUP_SIZE, group, 		1, 
												GROUP_SIZE, WARPS_PER_BLOCK, 1, 128, 0, cuPara, 0); /**/

	if (status != CUDA_SUCCESS){ fprintf(stderr, "cuLaunchKernel failed:%d:%d\n", __LINE__, status);	goto Error;	}

	///full copy...
	status = cuMemcpyDtoH(tData_GPU->dDistBig, tData_GPU->cu_dDistBig, con_grp*GROUP_SIZE * GROUP_SIZE * sizeof(FLOAT_T));
	if (status != CUDA_SUCCESS){ fprintf(stderr, "cuMemcpyDtoH failed: %20s %5d ,error:%5d\n", __FILE__, __LINE__, status);	goto Error;	}

	///OPT writing...
/*	for (k = 0; k < con_grp; k++){
		FLOAT_T *dd = tData_GPU->dDist[k];
//TODO: congroup things fix
		int jGroup = gj+k*threadNum;
		for(i = 0; i < GROUP_SIZE; i++){
			for(j = 0; j < GROUP_SIZE; j++){
				if(gi != jGroup || (gi == jGroup && i > j)){
					distElem[locDistElemI].i = gi*GROUP_SIZE+i;
					distElem[locDistElemI].j = jGroup*GROUP_SIZE+j;
					distElem[locDistElemI].dDist = dd[i * GROUP_SIZE + j];
					fprintf(fhandle, "%10.2f %d %d\n", distElem[locDistElemI].dDist, distElem[locDistElemI].i, distElem[locDistElemI].j);
					locDistElemI++;
				}
			}
		}
	}/**/

	for (k = 0; k < con_grp; k++){
		int jGroup = gj+k*threadNum;
		for(i = 0; i < GROUP_SIZE; i++){
			for(j = 0; j < GROUP_SIZE; j++){
				if(gi != jGroup || (gi == jGroup && i > j)){
					fprintf(fhandle, "%10.2f %d %d\n", tData_GPU->dDistBig[k*GROUP_SIZE*GROUP_SIZE+i*GROUP_SIZE+j], 
							gi*GROUP_SIZE+i, jGroup*GROUP_SIZE+j);
				}
			}
		}
	}/**/

//	fprintf(fhandle, "Rank %5d: Kernelrun Time:%10.2f sec\n",rank, getlocal_time());

	return CUDA_SUCCESS;
	Error: return status;
}

int resizeData(t_Data *cpuData, int newN){
	int result = 0;
	int nN = cpuData->nSeq;
	int nM = cpuData->nMaxLen;

	char *new_acSequences = (char*)malloc(sizeof(char)*nM*newN);
	int *new_anLen = (int*)malloc(sizeof(int)*newN);

	if (new_acSequences == NULL || new_anLen == NULL){
		fprintf(stderr, "error:%s,%s,%d\n", __FILE__, __FUNCTION__, __LINE__);
		result = -1;
		goto exit;
	}

	memset(new_acSequences, 0, nM*newN * sizeof(char));
	memset(new_anLen, 0, newN * sizeof(int));
	memcpy ( new_acSequences, cpuData->acSequences, nN*nM * sizeof(char) );
	memcpy ( new_anLen, cpuData->anLen, nN*sizeof(int) );

	free(cpuData->acSequences);
	free(cpuData->anLen);
	cpuData->acSequences = new_acSequences;
	cpuData->anLen = new_anLen;
	cpuData->nSeq = newN;
	exit: return result;
}
int pack_dist(t_mpi_dist* distElem, int elemNum, FLOAT_T *dDist, int nDistSize,
		int init, int orgDistSize){
	int result = 0;
	if (distElem == NULL || dDist == NULL){
		fprintf(stderr, "error:%s,%s,%d\n", __FILE__, __FUNCTION__, __LINE__);
		result = -1;
		goto exit;
	}

	//we need init the dDsit array
	if ( init == 1 ){
		memset(dDist, 0, nDistSize * sizeof(FLOAT_T));
	}
	for (int k = 0; k < elemNum; k++){
		int i = distElem[k].i;
		int j = distElem[k].j;
		int index = (i * i - i + 2 * j) >> 1;
		if(index < orgDistSize)
			dDist[index] = distElem[k].dDist;
	}
	exit: return result;
}

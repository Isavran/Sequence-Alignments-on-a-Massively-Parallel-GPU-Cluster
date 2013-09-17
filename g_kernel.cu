/**
 * g_kernel.cu
 *
 *  Created on: Dec 15, 2012
 *      Author: Ibrahim Savran
 *     Referenced ygao's Kernel
 */

#include <stdio.h>
#include "g_interface.h"

#define MAX_LEN 	801
#define MAXL 		800
#define FMAT_LINES 	2

#define INDEX_FM(a,b) 	(((stride+1)*((a)%FMAT_LINES)+(b))*GROUP_SIZE + tid)
#define INDEX_A(a) 		((bid * stride + (stride * groupA * GROUP_SIZE))+(a))
#define INDEX_B(a) 		((a)*GROUP_SIZE+(tid + (stride * groupB * GROUP_SIZE)))

///CONCURRENT_GROUPS
extern "C" __global__

void needlemanWunsch_CU(
		FLOAT_T *adLookup,
		const char* acA,
		const char* acB,
		int gi,					///
		int gj,					///
		int *anLen,
		FLOAT_T *dDistBig,
		int stride,				/// blokdimx=32; y =1 	tready 0
		int threadNum ) {
	int 	tid  = threadIdx.x;
	int 	bid  = blockIdx.x;				/// 
	int 	bgrp = blockIdx.y; 				/// .
	int 	groupA = gi;
	int 	groupB = gj + /*tgrp * threadNum */ + bgrp * threadNum /* *blockDim.y*/;	/// block dim 256 1 1


	int 	nLenA =  anLen[groupA * GROUP_SIZE + bid];
	int 	nLenB =  anLen[groupB * GROUP_SIZE + tid];

	if ( nLenA == 00 ){
		if ( 0 == nLenB )
			dDistBig[ bgrp * GROUP_SIZE * GROUP_SIZE+bid*GROUP_SIZE+tid ] = 0.0;
		else 	dDistBig[ bgrp * GROUP_SIZE * GROUP_SIZE+bid*GROUP_SIZE+tid ] = GAP_PENALTY; ///dDist[bid*GROUP_SIZE+tid] = GAP_PENALTY;
	} else {
		__syncthreads();

#ifdef SHARE_USAGE
		__shared__ int cache_lo[N_BASES*N_BASES];
		__shared__ int cache_hi[N_BASES*N_BASES];
			if (tid < N_BASES*N_BASES ){
				cache_lo[tid] = __double2loint(adLookup[tid]);
				cache_hi[tid] = __double2hiint(adLookup[tid]);
			}
#endif
		FLOAT_T aadFMatrix[MAXL];	
		short 	dDist[MAXL];			
		short j;
			for ( j = 0; j <= nLenB; j++ ){
				aadFMatrix[j] 	= GAP_PENALTY * j; 
				dDist[j]			= 0;
			}

			FLOAT_T laadFM;
			short left;
			for ( short i = 1; i <= nLenA; i++ ){
				laadFM = GAP_PENALTY * i;///0 a yaz	
				char acm = acA[ INDEX_A(i - 1) ];
				char ai = acm;
				left = 0;
				#pragma unroll 1
				for (j = 1; j <= nLenB; j++){
					FLOAT_T myd1, myd2, myd3;
					char bj =	acB[INDEX_B(j - 1)];
///================================================
#ifdef SHARE_USAGE	
					myd1 = aadFMatrix[j-1] + __hiloint2double(cache_hi[ ai*N_BASES + bj],
																cache_lo[ ai*N_BASES + bj]);
#else
					myd1 = aadFMatrix[j-1] + adLookup[ ai*N_BASES + bj];
#endif
					myd3 = aadFMatrix[j] + GAP_PENALTY;	/// up
					myd2 = laadFM + GAP_PENALTY;		/// left
					if( acm == bj ){
						myd2 += (HOMOPOLYMER_PENALTY - GAP_PENALTY);
						myd3 += (HOMOPOLYMER_PENALTY - GAP_PENALTY);
					}

					if (i == nLenA-1 )	/**/	myd2 = laadFM;///aadFMatrix[adr];
					if (j == nLenB-1 )			myd3 = aadFMatrix[j];///
				
					aadFMatrix[j-1] = laadFM;

					acm =  myd1 < myd2? DIAG: LEFT;		///dChoice1 		= fmin(dChoice1, dChoice2);
					if (acm == LEFT) myd1 = myd2;
					acm = myd1 < myd3 ? acm:UP;			///dChoice1 < dChoice2 ? : sAll[0][otid]	= LEFT;
					if (acm ==UP) myd1 = myd3;
					laadFM = myd1;	 

					short now;
					if 	(acm == DIAG) { 
						now = dDist[j-1]+1;
						acm = ai;
						if(ai != bj)		acm = -1;
						///else				acm = -1;
					}
					else if (acm == LEFT)  { acm = bj; now = left+1; 		if (i==nLenA-1)  now--;}
					else /*if (acm == UP)*/{ acm = ai; now = dDist[j]+1; 	if (j==nLenB-1)  now--;} 
					dDist[j-1] 	= left;
					left		= now;

///-------------------------------------------------
			}	///for j
			aadFMatrix[j-1] = laadFM;
			dDist[j-1]=left;
		}	/// for i
	dDistBig[ bgrp * GROUP_SIZE * GROUP_SIZE+bid * GROUP_SIZE+tid ] = laadFM/((FLOAT_T) left);
	}
}///End of the Kernel


/** This software and documentation is copyright © 2009 by Christopher Quince.*/

/** Permission is granted for anyone to copy, use, or modify these programs and
 * documents for purposes of research or education, provided this copyright
 * notice is retained, and note is made of any changes that have been made.*/

/** These programs and documents are distributed without any warranty, express
 * or implied. As the programs were written for research purposes only, they have
 * not been tested to the degree that would be advisable in any important application.
 * All use of these programs is entirely at the user's own risk.*/

/**System includes****/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include <pthread.h>

#include "SeqDist.h"
#include "g_interface.h"
FILE *glb_outFiles[10];

int needlemanWunschGrp(int gi, int gj, int iSz, int jSz, t_mpi_dist* distElem,
		int distElemI, t_Data *tData);
int needlemanWunschThread(t_cu_Thread *cu_Thread);

/**global constants*/
static char *usage[] =
{ "SeqDist - pairwise distance matrix from a fasta file\n",
		"-in     string            fasta file name\n", "Options:\n",
		"-i output identifiers \n",
		"-rin    string            lookup file name\n" };

static int nLines = 5;
static char szSequence[] = "ACGT";	///
static FLOAT_T* adLookUp = NULL;
void broadcastData(t_Data *ptData){
	int i = 0;
	MPI_Bcast((void *) &ptData->nSeq, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) &ptData->nMaxLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) ptData->anLen, ptData->nSeq, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) ptData->acSequences, ptData->nSeq * ptData->nMaxLen, MPI_CHAR, 0, MPI_COMM_WORLD);
}

void receiveData(t_Data *ptData){
	MPI_Bcast((void *) &ptData->nSeq, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) &ptData->nMaxLen, 1, MPI_INT, 0, MPI_COMM_WORLD);

	ptData->anLen = (int *) malloc(ptData->nSeq * sizeof(int));
	if (!ptData->anLen)
		goto memoryError;

	MPI_Bcast((void *) ptData->anLen, ptData->nSeq, MPI_INT, 0, MPI_COMM_WORLD);
	ptData->acSequences = (char *) malloc(
			ptData->nSeq * ptData->nMaxLen * sizeof(char));
	if (!ptData->acSequences)
		goto memoryError;
	MPI_Bcast((void *) ptData->acSequences, ptData->nSeq * ptData->nMaxLen,
			MPI_CHAR, 0, MPI_COMM_WORLD);
	return;
	memoryError: fprintf(stderr, "Failed allocating memory in receiveData\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]){
	int a = 0, i = 0, j = 0, nN = 0, nM = 0;
	t_Params tParams;
	t_Data tData;
	FLOAT_T *adDists = NULL;
	t_mpi_dist *distElem = NULL;
	int distElemI = 0;
	int numtasks, rank, rc;
	int offset, nA, nA0, nSize, nCount, nStart, nFinish, nTag = 1;
	int nOrgSize;
	int nPackets = 0, nPacketSize = 0, nPacketCurr = 0;
	int totalSz, incSZ;
	int elemNum;
	int endSz;
	int gNum, nRem, gi, gj;
	int *mpiPckLen = NULL;
	int orgN;

	MPI_Status status;
	///GPU and thread
	t_cu_Thread cu_Thread[GPU_NUM];
	int iThread[GPU_NUM];
	pthread_t thread[GPU_NUM];

	fflush(stdout);
	rc = MPI_Init(&argc, &argv);	///
	if (rc != MPI_SUCCESS){
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);		///how many 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);			///read the rank
	/** get command line params*/
	getCommandLineParams(&tParams, argc, argv);		///read parameters
	fflush(stdout);
	if (rank == 0){
		///start timer
		start_time();
		startlocal_time();
	/**head node reads data*/
		///printf("%d read data\n",rank); fflush(stdout);
		initLookUp(&tParams);
		readData(&tData, &tParams);
		MPI_Bcast(adLookUp, N_BASES * N_BASES, SD_MPI_FP_TYPE, 0, MPI_COMM_WORLD);
		///printf("%d broadcast data\n",rank); fflush(stdout);
		//Init_time = get_time();
		//fprintf(stderr, "ROOT: Read Data Time:%10.2f sec\n", Init_time );
		fprintf(stderr, "Rank %5d: Read data Time:\t  %10.2f sec\n",rank, get_time());
		/**do my bit*/
		nN = tData.nSeq;
		nM = tData.nMaxLen;
		gNum = nN / GROUP_SIZE;
		/**allocate memory for whole dist matrix*/
		nOrgSize = (nN * (nN - 1)) / 2;
		orgN = nN;
/* Not needed
#ifdef GPU_ACC_ON
		///pack sequence number in group
		
		if( nN % (GROUP_SIZE*CONCURRENT_GROUPS) != 0){
			gNum = ((nN / (GROUP_SIZE* CONCURRENT_GROUPS )) + 1)
					*CONCURRENT_GROUPS;
		}
///			gNum++;
		nN = GROUP_SIZE * gNum;

#endif /**/

		/**realloc for memory GROUP_SIZE align*/
		resizeData(&tData, nN);
		nSize = (nN * (nN - 1)) / 2;
		/**sends it out to other nodes*/
		broadcastData(&tData);
		/// the unaligned sequence number
		fprintf(stderr, "Rank %5d: Cuda Broadcast Time:\t %10.2f sec\n",rank, get_time());
		nRem = nN - gNum * GROUP_SIZE;
		/// relative large for nA
		nA = (nSize / numtasks) << 1;
		printf("%d\n", orgN);
		/// TODO: adDist may not hold all the values at one time malloc
		/// adDists become large enough to hold all the values
		adDists = (FLOAT_T *) malloc(nOrgSize * sizeof(FLOAT_T));
		if (!adDists)	goto memoryError;
		/// alloc distElem for pack
		distElem = (t_mpi_dist *) malloc(nA * sizeof(t_mpi_dist));
		if (!distElem)	goto memoryError;
		distElemI = 0;
		/// alloc mpi package length array
		mpiPckLen = (int *) malloc(nA * sizeof(mpiPckLen));
		if (!mpiPckLen)
			goto memoryError;
		/// for pthread
		for (i = 0; i < GPU_NUM; i++){
			cu_Thread[i].gNum = gNum;
			cu_Thread[i].rank = rank;
			cu_Thread[i].numtasks = numtasks;
			cu_Thread[i].threadId = i;
			cu_Thread[i].threadNum = GPU_NUM;
			cu_Thread[i].tData = &tData;
			cu_Thread[i].tData_GPU = (t_Data_GPU*) malloc(sizeof(t_Data_GPU));
			/// thread need to fill these by them self
//			cu_Thread[i].distElem = (t_mpi_dist *) malloc( sizeof(t_mpi_dist) * (nA / GPU_NUM + 1));
//			cu_Thread[i].distElemI = 0;
			cu_Thread[i].adLookUp = adLookUp;
			iThread[i] = pthread_create(&thread[i], NULL, needlemanWunschThread, (void*) (&cu_Thread[i]));
		}
		for (i = 0; i < GPU_NUM; i++){
			pthread_join(thread[i], NULL);
			memcpy(&distElem[distElemI], cu_Thread[i].distElem,	cu_Thread[i].distElemI * sizeof(t_mpi_dist));
			distElemI += cu_Thread[i].distElemI;
		}	///gpu num

		fprintf(stderr, "Time:%10.2f sec\n", get_time());
	}
	else{
		///TODO: more than one process fix on C005_s60_c01_T220_320.fa
		/** receive data*/
		startlocal_time();
		adLookUp = (FLOAT_T *) malloc(N_BASES * N_BASES * sizeof(FLOAT_T));
		if (!adLookUp)
			goto memoryError;
		MPI_Bcast(adLookUp, N_BASES * N_BASES, SD_MPI_FP_TYPE, 0, MPI_COMM_WORLD);
		receiveData(&tData);	///read the data
		nN = tData.nSeq;
		nM = tData.nMaxLen;
		nSize = (nN * (nN - 1)) / 2;
		///sequence number in group
		gNum = nN / GROUP_SIZE;
		///the unaligned sequence number
		nRem = nN - gNum * GROUP_SIZE;
		///relative large for nA
		nA = (nSize / numtasks) << 1;
		///alloc distElem for pack
		distElem = (t_mpi_dist *) malloc(nA * sizeof(t_mpi_dist));
		if (!distElem)
			goto memoryError;
		distElemI = 0;
		/// for pthread
		for (i = 0; i < GPU_NUM; i++){
			cu_Thread[i].gNum = gNum;
			cu_Thread[i].rank = rank;
			cu_Thread[i].numtasks = numtasks;
			cu_Thread[i].threadId = i;
			cu_Thread[i].threadNum = GPU_NUM;
			cu_Thread[i].tData = &tData;
			//thread need to fill these by them self
//			cu_Thread[i].distElem = (t_mpi_dist *) malloc( sizeof(t_mpi_dist) * (nA / GPU_NUM + 1) );
//			cu_Thread[i].distElemI = 0;
			cu_Thread[i].adLookUp = adLookUp;
			iThread[i] = pthread_create(&thread[i], NULL, needlemanWunschThread, (void*) (&cu_Thread[i]));
		}
		for (i = 0; i < GPU_NUM; i++){
			pthread_join(thread[i], NULL);
		//	memcpy(&distElem[distElemI], cu_Thread[i].distElem, cu_Thread[i].distElemI * sizeof(t_mpi_dist));
		//	distElemI += cu_Thread[i].distElemI;
		}
		///send the package size first
///deleted
	}

	/**free allocated memory*/
	free(adDists);
	free(tData.acSequences);
	free(tData.anLen);
	/// Wait for everyone to stop
	MPI_Barrier(MPI_COMM_WORLD);
	/// Always use MPI_Finalize as the last instruction of the program
	MPI_Finalize();
	exit(EXIT_SUCCESS);
	memoryError: fprintf(stderr, "Failed allocating memory in main\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
}

void writeUsage(FILE* ofp){
	int i = 0;
	char *line;
	for (i = 0; i < nLines; i++){
		line = usage[i];
		fputs(line, ofp);
	}
}

char *extractParameter(int argc, char **argv, char *param, int when){
	int i = 0;
	while ((i < argc) && (strcmp(param, argv[i])))	i++;
	if (i < argc - 1)	return (argv[i + 1]);

	if ((i == argc - 1) && (when == OPTION))	return "";
	if (when == ALWAYS)
		fprintf(stdout, "Can't find asked option %s\n", param);

	return (char *) NULL;
}

void getCommandLineParams(t_Params *ptParams, int argc, char *argv[]){
	char *szTemp = NULL;
	char *cError = NULL;

	/**get parameter file name*/
	ptParams->szInputFile = extractParameter(argc, argv, INPUT_FILE, ALWAYS);
	if (ptParams->szInputFile == NULL)	goto error;

	if (szTemp = extractParameter(argc, argv, LOOKUP_FILE_FLAG, OPTION))
		ptParams->szLookUpFile = szTemp;
	else{
		ptParams->szLookUpFile = getenv("SEQ_LOOKUP_FILE");
		if (ptParams->szLookUpFile == NULL)		ptParams->szLookUpFile = LOOKUP_FILE;
		
	}
	/**identifiers*/
	szTemp = extractParameter(argc, argv, IDENT, OPTION);
	if (szTemp != NULL)	ptParams->bIdent = TRUE;
	else	ptParams->bIdent = FALSE;

	szTemp = extractParameter(argc, argv, PHYLIP, OPTION);
	if (szTemp != NULL)		ptParams->bPhylip = TRUE;
	else		ptParams->bPhylip = FALSE;
	return;

	error: writeUsage(stdout);
	exit(EXIT_FAILURE);
}

FLOAT_T dist(char cA, char cB){
	int nA = 0, nB = 0;
	switch (cA){
	case 'A':	nA = 0;	break;
	case 'C':	nA = 1;	break;
	case 'T':	nA = 2;	break;
	case 'G':	nA = 3;	break;
	default:
		fprintf(stderr, "Non standard base %c\n", cA);
		return 0.0;
	}
	switch (cB){
	case 'A':	nB = 0;	break;
	case 'C':	nB = 1;	break;
	case 'T':	nB = 2;	break;
	case 'G':	nB = 3;	break;
	default:
		fprintf(stderr, "Non standard base %c\n", cB);
		return 0.0;
	}
	return adLookUp[nA * N_BASES + nB];
}

FLOAT_T dmin3(FLOAT_T dA, FLOAT_T dB, FLOAT_T dC){
	FLOAT_T dAB = dA < dB ? dA : dB;
	return dC < dAB ? dC : dAB;
}

int getMove(FLOAT_T dA, FLOAT_T dB, FLOAT_T dC){

	if (dA < dB){
		if (dA < dC)return DIAG;
		else		return UP;
	}else{
		if (dB < dC)return LEFT;
		else		return UP;
	}
}

char getLastMatch(int nMove, int nI, int nJ, const char *acA, const char *acB){
	switch (nMove){
		case DIAG:	if (acA[nI - 1] == acB[nJ - 1])	return acA[nI - 1];
						else			return '\0';
		break;
		case LEFT:	return acB[nJ - 1];	break;
		case UP:	return acA[nI - 1];		break;
	}
}

void updateHomopolymers(int** aanHA, int** aanHB, int** aanMoves, int nI,
		int nJ, const char* acA, const char *acB){
	int nMove = aanMoves[nI][nJ];

	switch (nMove){
	case DIAG:
		if (acA[nI - 1] == acB[nJ - 1]){
			if (getLastMatch(aanMoves[nI - 1][nJ - 1], nI - 1, nJ - 1, acA, acB)
					== acA[nI - 1]){
				aanHA[nI][nJ] = aanHA[nI - 1][nJ - 1] + 1;
				aanHB[nI][nJ] = aanHB[nI - 1][nJ - 1] + 1;
			}else{
				aanHA[nI][nJ] = 1;
				aanHB[nI][nJ] = 1;
			}
		}else{
			aanHA[nI][nJ] = 0;
			aanHB[nI][nJ] = 0;
		}
		break;
	case LEFT: /*gap in a*/
		if (acB[nJ - 1] == getLastMatch(aanMoves[nI][nJ - 1], nI, nJ - 1, acA, acB)){
			aanHB[nI][nJ] = aanHB[nI][nJ - 1] + 1;
			aanHA[nI][nJ] = aanHA[nI][nJ - 1] + 1;
		}else{
			aanHA[nI][nJ] = 0;
			aanHB[nI][nJ] = 0;
		}
		break;

	case UP:	/*GAP in B*/
		if (acA[nI - 1] == getLastMatch(aanMoves[nI - 1][nJ], nI - 1, nJ, acA, acB)){
			aanHA[nI][nJ] = aanHA[nI - 1][nJ] + 1;
			aanHB[nI][nJ] = aanHB[nI - 1][nJ] + 1;
		}else{
			aanHA[nI][nJ] = 0;
			aanHB[nI][nJ] = 0;
		}
		break;
	}
}

int returnHomopolymerA(int nMove, int** aanHA, int** aanMoves, int nI, int nJ,
		const char* acA, const char *acB){
	int retA = 0;

	switch (nMove){
	case DIAG:
		if (acA[nI - 1] == acB[nJ - 1]){
			if (getLastMatch(aanMoves[nI - 1][nJ - 1], nI - 1, nJ - 1, acA, acB)
					== acA[nI - 1]){
				retA = aanHA[nI - 1][nJ - 1] + 1;
			}else{
				retA = 1;
			}
		}else{
			retA = 0;
		}

		break;
	case LEFT:
		if (acB[nJ - 1]
				== getLastMatch(aanMoves[nI][nJ - 1], nI, nJ - 1, acA, acB)){
			retA = aanHA[nI][nJ - 1] + 1;
		}else{
			retA = 0;
		}
		break;

	case UP:
		/*GAP in B*/
		if (acA[nI - 1]
				== getLastMatch(aanMoves[nI - 1][nJ], nI - 1, nJ, acA, acB)){
			retA = aanHA[nI - 1][nJ] + 1;
		}else	{
			retA = 0;
		}
		break;
	}
	return retA;
}

FLOAT_T needlemanWunsch(const char* acA, const char* acB, int nLenA, int nLenB,
		int nM){
	FLOAT_T **aadFMatrix = NULL;
	int **aanMoves = NULL;
	int **aanHA = NULL;
	int **aanHB = NULL;
	int anHA[nLenA + nLenB];int
	anHB[nLenA + nLenB];char
	*acAlignA = NULL, *acAlignB = NULL;
	int nCount = 0, nLen = 0, nComp = 0;
	int i = 0, j = 0;
	FLOAT_T dDist = 0.0;

	aadFMatrix = (FLOAT_T **) malloc((nLenA + 1) * sizeof(FLOAT_T *));
	aanMoves = (int **) malloc((nLenA + 1) * sizeof(int *));
	aanHA = (int **) malloc((nLenA + 1) * sizeof(int *));
	aanHB = (int **) malloc((nLenA + 1) * sizeof(int *));
	if (!aadFMatrix || !aanMoves)	goto memoryError;

	for (i = 0; i < nLenA + 1; i++){
		aadFMatrix[i] = (FLOAT_T *) malloc((nLenB + 1) * sizeof(FLOAT_T));
		aanMoves[i] = (int *) malloc((nLenB + 1) * sizeof(int));
		aanHA[i] = (int *) malloc((nLenB + 1) * sizeof(int));
		aanHB[i] = (int *) malloc((nLenB + 1) * sizeof(int));
		if (!aadFMatrix[i] || !aanMoves[i])
			goto memoryError;

		for (j = 0; j < nLenB + 1; j++){
			aadFMatrix[i][j] = 0.0;
			aanMoves[i][j] = -1;
			aanHA[i][j] = -1;
			aanHB[i][j] = -1;
		}
	}

	for (i = 0; i <= nLenA; i++){
		aadFMatrix[i][0] = GAP_PENALTY * i;
		aanMoves[i][0] = UP;
		aanHA[i][0] = 0;
		aanHB[i][0] = 0;
	}
	for (j = 0; j <= nLenB; j++){
		aadFMatrix[0][j] = GAP_PENALTY * j;
		aanMoves[0][j] = LEFT;
		aanHA[0][j] = 0;
		aanHB[0][j] = 0;
	}
///	fprintf(stderr, "cpu start\n");
	for (i = 1; i <= nLenA; i++){
		for (j = 1; j <= nLenB; j++){
			FLOAT_T dChoice1, dChoice2, dChoice3;
			dChoice1 = aadFMatrix[i - 1][j - 1] + dist(acA[i - 1], acB[j - 1]);
			if (i == nLenA){
				dChoice2 = aadFMatrix[i][j - 1];
			}
			else{
				FLOAT_T dGap = 0.0;
///				int nCurrH = aanHA[i][j - 1];
				int nNewH = returnHomopolymerA(LEFT, aanHA, aanMoves, i, j, acA, acB);
				/**Left gap in A*/
				if (nNewH == 0){
					dGap = GAP_PENALTY;
				}else{
					dGap = HOMOPOLYMER_PENALTY;
				}

				dChoice2 = aadFMatrix[i][j - 1] + dGap;
			}

			if (j == nLenB){
				dChoice3 = aadFMatrix[i - 1][j];
			}else	{
				FLOAT_T dGap = 0.0;
///				int nCurrH = aanHA[i][j - 1];
				int nNewH = returnHomopolymerA(LEFT, aanHA, aanMoves, i, j, acA, acB);
				/**Left gap in A*/
				if (nNewH == 0){
					dGap = GAP_PENALTY;
				}else{
					dGap = HOMOPOLYMER_PENALTY;
				}
				/**Up gap in B*/
				dChoice3 = aadFMatrix[i - 1][j] + dGap;
			}
			aanMoves[i][j] = getMove(dChoice1, dChoice2, dChoice3);
			aadFMatrix[i][j] = dmin3(dChoice1, dChoice2, dChoice3);
			updateHomopolymers(aanHA, aanHB, aanMoves, i, j, acA, acB);
///			if(i<=32 && j <= 32)
///			fprintf(stderr, "%10.2f ", aadFMatrix[i][j]);
///				fprintf(stderr, "%10c ", acB[j - 1]);
		}
///		if(i<=32)
///		fprintf(stderr, "\n");
	}
///	fprintf(stderr, "cpu end\n");
	dDist = aadFMatrix[nLenA][nLenB];

	acAlignA = (char *) malloc((nLenA + nLenB) * sizeof(char));
	if (!acAlignA)
		goto memoryError;

	acAlignB = (char *) malloc((nLenA + nLenB) * sizeof(char));
	if (!acAlignB)
		goto memoryError;

	for (i = 0; i < nLenA + nLenB; i++){
		acAlignA[i] = GAP;
		acAlignB[i] = GAP;
	}

	nCount = 0;
	i = nLenA;
	j = nLenB;
	while (i > 0 && j > 0){
		switch (aanMoves[i][j]){
		case DIAG:
			acAlignA[nCount] = acA[i - 1];
			acAlignB[nCount] = acB[j - 1];
			i--;
			j--;
			break;
		case UP:
			acAlignA[nCount] = acA[i - 1];

			if (j == nLenB){
				acAlignB[nCount] = T_GAP;
			}
			else{
				acAlignB[nCount] = GAP;
			}
			i--;
			break;
		case LEFT:
			if (i == nLenA){
				acAlignA[nCount] = T_GAP;
			}
			else{
				acAlignA[nCount] = GAP;
			}
			acAlignB[nCount] = acB[j - 1];

			j--;
			break;
		}
		nCount++;
	}

	while (i > 0){
		acAlignA[nCount] = acA[i - 1];
		acAlignB[nCount] = GAP;

		i--;
		nCount++;
	}

	while (j > 0){
		acAlignA[nCount] = GAP;
		acAlignB[nCount] = acB[j - 1];

		j--;
		nCount++;
	}
	nLen = nCount;
	i = 0;
	while (acAlignA[i] == T_GAP || acAlignB[i] == T_GAP){
		i++;
	}

	nComp = nLen - i;

	dDist = aadFMatrix[nLenA][nLenB] / ((FLOAT_T) nComp); //normalise by M * true length not with terminal gaps

	free(acAlignA);
	free(acAlignB);
	for (i = 0; i <= nLenA; i++){
		free(aadFMatrix[i]);
		free(aanMoves[i]);
		free(aanHA[i]);
		free(aanHB[i]);
	}
	free(aadFMatrix);
	free(aanMoves);
	free(aanHA);
	free(aanHB);
	return dDist;

	memoryError: fprintf(stderr,
			"Failed to allocate memory in needlemanWunsch\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
}

int needlemanWunschGrp(int gi, int gj, int iSz, int jSz, t_mpi_dist* distElem,
		int distElemI, t_Data *tData){
	int i, j;
	int iStart = gi * GROUP_SIZE;
	int jStart = gj * GROUP_SIZE;
	FLOAT_T dDist;
	int locDistElemI = distElemI;

	fflush(stderr);
	for (i = iStart; i < iStart + iSz; i++){
		for (j = jStart; j < jStart + jSz; j++){
			if (i > j){
//				if(i==32 && j==0)
//				{
				dDist = needlemanWunsch(&tData->acSequences[i * tData->nMaxLen],
						&tData->acSequences[j * tData->nMaxLen],
						tData->anLen[i], tData->anLen[j], tData->nMaxLen);
//				fprintf(stderr, "%10.2f \n", dDist);
//				}
				distElem[locDistElemI].i = i;
				distElem[locDistElemI].j = j;
				distElem[locDistElemI].dDist = dDist;
				locDistElemI++;
			}
		}
	}
	return 0;
}

int needlemanWunschThread(t_cu_Thread *cu_Thread){
	int result = 0;
	int gi, gj;
	int gNum = cu_Thread->gNum;
	int rank = cu_Thread->rank;
	int numtasks = cu_Thread->numtasks;
	int threadNum = cu_Thread->threadNum;
	int threadId = cu_Thread->threadId;
	int distElemI = 0;
	t_mpi_dist *distElem = cu_Thread->distElem;
	t_Data *tData = cu_Thread->tData;
	
	tData->allIndex = (int *) malloc(tData->nSeq * sizeof(int)); ///allindex yer acildi
	tData->Allmax = (FLOAT_T*) malloc(tData->nSeq * sizeof(FLOAT_T)); ///allindex yer acildi
	int *allIndex = tData->allIndex;	
	char outFileName[500];
	startlocal_time();///	 for cuda init
	sprintf(outFileName, "out_%d_%d.txt", rank, threadId);
	glb_outFiles[threadId] = fopen(outFileName, "w");
#ifdef GPU_ACC_ON
	//alloc the GPU resource holder
	cu_Thread->tData_GPU = (t_Data_GPU*) malloc(sizeof(t_Data_GPU));
	if (cu_Thread->tData_GPU == NULL){ fprintf(stderr, "error:%s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);	result = -1;}

	if (initCUDA(cu_Thread) != CUDA_SUCCESS){
		fprintf(stderr, "error:%s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);	result = -1;}

	if (cuResAlloc(cu_Thread, glb_outFiles[threadId]) != CUDA_SUCCESS){
		fprintf(stderr, "error:%s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);	result = -1;}
#endif
	
// TODO:CONCURRENT_GROUPS fix
	for (gi = 0; gi < gNum; gi++){
		gj = 0;
		if (gi % numtasks == rank){
				int con_grp = CONCURRENT_GROUPS;
#ifdef GPU_ACC_ON
				con_grp = (gi-gj)/threadNum + 1;
			///Call all the at a time
			cuNeedlemanWunsch(gi, gj, cu_Thread, distElem, distElemI, con_grp);
#else
			needlemanWunschGrp(gi, gj, GROUP_SIZE, GROUP_SIZE, distElem, distElemI, tData);
#endif
		}//if gi
	}///for gi
	//exit(0);
	cu_Thread->distElemI = distElemI;
	return result;
}
///---------------------------------------------------------------------
void readData(t_Data *ptData, t_Params *ptParams){	///
	FILE *ifp = NULL;
	char szLine[MAX_LINE_LENGTH];
	int nPos = 0, i = 0, j = 0, nM = 0, nSequences = 0;
	char *szBrk;	///break
	char *szRet;	///return
	/** first count sequences and get length*/
	ptData->nSeq = 0;	
	ptData->nMaxLen = 0;
	ifp = fopen(ptParams->szInputFile, "r");
	if (ifp){
		while (fgets(szLine, MAX_LINE_LENGTH, ifp)){
			if (szLine[0] == '>'){
				if (nPos > ptData->nMaxLen){
					ptData->nMaxLen = nPos;
				}
				ptData->nSeq++;
				nPos = 0;
			}
			else{
				i = 0;
				while (strrchr(szSequence, szLine[i]) != NULL){
					i++;
					nPos++;
				}
			}
		}
		fclose(ifp);
	} else{
		fprintf(stderr, "Can't open input file %s\n", ptParams->szInputFile);
		exit(EXIT_FAILURE);
	}
	ptData->aszID = (char **) malloc(ptData->nSeq * sizeof(char *));
	if (nPos > ptData->nMaxLen){
		ptData->nMaxLen = nPos;
	}
	nM = ptData->nMaxLen;
	ptData->acSequences = (char *) malloc(ptData->nSeq * nM * sizeof(char));
	ptData->anLen = (int *) malloc(ptData->nSeq * sizeof(int));

	ifp = fopen(ptParams->szInputFile, "r");
	if (ifp){
		while (szRet = fgets(szLine, MAX_LINE_LENGTH, ifp)){
			if (szLine[0] == '>'){
				if (nSequences > 0){
					ptData->anLen[nSequences - 1] = nPos;
				}
				szBrk = strpbrk(szLine, " \n");		///put break...
				(*szBrk) = '\0';
				ptData->aszID[nSequences] = strdup(szLine + 1);
				nPos = 0;
				nSequences++;
			}
			i = 0;
			while (szLine[i] != '\0' && strrchr(szSequence, szLine[i]) != NULL){ 		///szSeq = ACTG 
				ptData->acSequences[(nSequences - 1) * nM + nPos] = toupper(szLine[i]); ///one char read
				nPos++;
				i++;
			}
		}
		ptData->anLen[nSequences - 1] = nPos;
		fclose(ifp);
	}
	else{
		fprintf(stderr, "Can't open input file %s\n", ptParams->szInputFile);
		exit(EXIT_FAILURE);
	}
}

void initLookUp(t_Params *ptParams){
	int i = 0, j = 0;
	FILE *ifp = NULL;
	adLookUp = (FLOAT_T *) malloc(N_BASES * N_BASES * sizeof(FLOAT_T));
	if (!adLookUp)
		goto memoryError;

	ifp = fopen(ptParams->szLookUpFile, "r");
	if (ifp){
		char szLine[MAX_LINE_LENGTH];
		char *pcError = NULL;
		char *szRet = NULL;
		char *szTok = NULL;

		for (i = 0; i < N_BASES; i++){
			fgets(szLine, MAX_LINE_LENGTH, ifp);

			szRet = strpbrk(szLine, "\n");
			(*szRet) = '\0';
			for (j = 0; j < N_BASES; j++){
				if (j == 0){
					szTok = strtok(szLine, COMMA);
				}else{
					szTok = strtok(NULL, COMMA);
				}
				adLookUp[i * N_BASES + j] = strtod(szTok, &pcError);
				if (*pcError != '\0')
					goto formatError;
			}
		}
		fclose(ifp);
	}
	else{
		fprintf(stderr, "Failed to open %s\n", ptParams->szLookUpFile);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
	return;
	memoryError: fprintf(stderr, "Failed allocating memory in initLookUp\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	formatError: fprintf(stderr, "Format error LookUp.dat\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
}

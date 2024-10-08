/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <unistd.h>
#include <limits.h>
#include <float.h>
#include <ammintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define BITS      64
#define SIMD_BITS 256
#define MAX_IMPS  50000000
#define SAMPLE_SZ 2048
#define REPETITION  10000   /* how many times to repeat each experiment to get an avg */
#define BITRANGE    1    /* 1 = search by bin. 0 = by sampling the data*/

/* printing options */
#define STATS          if (0)
#define VERBOSE        if (1)
#define PRINT_HISTO    if (0)
#define PRINT_IMPRINTS if (0)
#define PRINT_QUERIES  if (0)

#undef COARSE
#ifdef COARSE
#define PAGESIZE sysconf(_SC_PAGESIZE) /* normal page size */
#else
#define PAGESIZE 64
#endif

/* Using MonetDB style types */
#define TYPE_void   0
#define TYPE_bte    3
#define TYPE_sht    4
#define TYPE_int    6
#define TYPE_oid    7
#define TYPE_flt    10
#define TYPE_dbl    11
#define TYPE_lng    12
#define TYPE_str    13

#define setBit(X,Y)      ((X) | ((unsigned long long) 1 << (Y)))
#define isSet(X,Y)       (((((unsigned long long)1)<<Y) & X) ? 1 : 0)
#define COMPRESSION_MASK (~((~((unsigned long) 0))<<(imps->bins)))
#define getMask(I)       (imps->bins==64?imps->imprints[I]:((((unsigned long) imps->imprints[((I)*imps->bins)/64])>>(((I)%(64/imps->bins))*(imps->bins))) & COMPRESSION_MASK))

#define GETBIT_GENERAL(Z, V, X)						\
	do {											\
		int _i;										\
		Z = 0;										\
		for (_i = 0; _i < imps->bins-1; _i++)		\
			Z += (V > imps->bounds[_i].X);			\
	} while (0)

typedef union {
	char bval;
	short sval;
	int ival;
	long lval;
	unsigned long ulval;
	float fval;
	double dval;
} ValRecord;

typedef struct {
	char          filename[1024];
	char          colname[1024];
	char          typename[64];
	int           coltype;			/* as given by TYPE_* defines	*/
	int           typesize;			/* size of type in bytes		*/
	unsigned long colcount;			/* count of values in column	*/
	char          *col;				/* heap of data					*/
	ValRecord     min;				/* min value in col				*/
	ValRecord     max;				/* max value in col				*/
	int           sorted;			/* 1 if sorted					*/
} Column;

/* The Column Imprints contains a binned imprint bitvector
 * to weed out blocks of no interest in a scan.
 * The imprint vectors are simply compressed, keeping track on
 * the number of blocks each imprint covers. From the blks we can
 * calculate the actual oid ranges, provided we scan only.
 */
#define MAXOFFSET 24
typedef struct {
	unsigned int   blks:MAXOFFSET;	/* blocks count						*/
	unsigned int   repeated:1;		/* same or unique imprints in blks	*/
	unsigned int   flgs:8 * sizeof(int) - MAXOFFSET-1; /* for future use	*/
} Dct;

typedef struct {
	Dct            *dct;		/* the dictionary structure				*/
	char           *imprints;	/* the imprint vectors					*/
	ValRecord      *bounds;		/* the bounds of the binning			*/
	unsigned long  dct_cnt;		/* count of dictionary entries			*/
	unsigned long  imps_cnt;	/* count of imprint vector entries		*/
	int            bins;		/* number of bins						*/
	int            imprintsize;	/* bytes per imprint					*/
	int            blocksize;	/* bytes per block						*/
} Imprints_index;

/* Zonemaps */
typedef struct {
	ValRecord min;
	ValRecord max;
} Zonemap;

typedef struct {
	Zonemap       *zmaps;
	unsigned long zmaps_cnt;
	int           zonesize; /* bytes per zone */
} Zonemap_index;

/* function declarations */
Imprints_index* create_imprints(Column *column, int blocksize, int max_bins, int simd);
Zonemap_index*  create_zonemaps(Column *column, int blocksize);
void queries(Column *column, Zonemap_index *zonemaps, Imprints_index *scalar_imps,
             Imprints_index *simd_imps, Imprints_index **exper_imps, FILE *qfile);

/* utils */
void isSorted(Column *column);
long usec();
__m256i setbit_256(__m256i x,int k);

/* helper functions */
void compareImprintsIndex(Column *column, Imprints_index *imps1, Imprints_index *imps2);
void printBounds(Column *column, Imprints_index *imps);
void printMask(char *mask, int byte);
void printImprint(Column *column, Imprints_index *imps);
void printBits(size_t const size, void const * const ptr);
/*
void printHistogram(long histo[BITS], char *name);
void printBins(Column column);
void printMask(long mask, int limit);
void printImprint(Column column, Zonemap_index *zonemaps);
void statistics(Column column);
*/

#ifdef __APPLE__
#define aligned_alloc(align, n) malloc(n)
#endif

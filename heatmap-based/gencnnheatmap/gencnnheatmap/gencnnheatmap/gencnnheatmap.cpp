#include "stdafx.h"
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<queue>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <highgui.h>
#include<opencv2/opencv.hpp>
#include<sstream>
#include<omp.h>
#include<hdf5.h>


int st, en;


//#define file96prefix "G:\\skinanddepth\\datasmall160withuv85size96_"
#define file96prefix "G:\\heatmap\\nyutest\\cnnheatmap"
#define N 8248

char file96[111];
#define DIML1 8248
#define DIML2 6776

using namespace std;
using namespace cv;
int nowcnt = 0;

float label[DIML1][DIML2];



Mat srcimg;
IplImage *img[3];
int all = 0;
int hsh[72758], sta[72758];

const int ll[11] = { 1, 7001, 14001, 21001, 28001, 35001, 42001, 49001, 56001, 63001, 70001 };
const int rr[11] = { 7000, 14000, 21000, 28000, 35000, 42000, 49000, 56000, 63000, 70000, 72756 };
FILE *fin = fopen("G:\\heatmap\\nyutest\\0heat.txt", "r");
void init()
{
	

	for (int i = st; i <=en; i++)
	{
		cout << i << "\n";
		for (int j = 0; j < 6776; j++)
			fscanf(fin, "%f", &label[i-st][j]);
	}
}

void genHDF5()
{
	//96*96
	hid_t fileid,  labelid, labelspaceid;
	herr_t status;
	fileid = H5Fcreate(file96, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	
	hsize_t dimslabel[2];
	

	dimslabel[0] = DIML1;
	dimslabel[1] = DIML2;


	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label96", H5T_IEEE_F32LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


	H5Dwrite(labelid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);


	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);


}
int main()
{
	
	for (int id = 0; id < 1; id++)
	{
		for (int i = 1; i <= 1; i++)
		{
			//72756/DIML1
			st =  (i - 1)*DIML1 + 1;
			en =  i*DIML1;
			init();
			nowcnt = 0;
			sprintf(file96, "%s%d%s", file96prefix, id * 72756 / DIML1 + i, ".h5");			
			genHDF5();

		}
	}
	fclose(fin);
	return 0;
}
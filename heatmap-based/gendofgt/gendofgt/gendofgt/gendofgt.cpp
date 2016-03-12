
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
#include<H5Cpp.h>
#define depprefix "H:\\libmodel\\dofgt\\"
#define depdir "I:\\libmodel\\deeppriordata\\"
#define N 72756
#define DIML1 9088
#define DIML2 55
#define SIZE 128
using namespace cv;
using namespace std;
using namespace H5;
int minh, maxh, minw, maxw;
float ave0;
float portion[111111];
float data[DIML1][1][SIZE][SIZE];
float label[DIML1][DIML2];
float all[N][DIML2];
char file[111];
int seq[N];
IplImage *img;
int st, en, nowcnt;
void init()
{
	FILE *findofgt = fopen("D:\\CNN\\PSO\\psogt2.txt", "r");
	for (int i = 0; i < 72756; i++)
	{
		int tmp;
		fscanf(findofgt, "%d", &tmp);
		for (int j = 0; j < 55; j++) fscanf(findofgt, "%f",&all[i][j] );
	}
		
	fclose(findofgt);
	
	FILE *fseq = fopen("D:\\CNN\\gendeeppriordata\\ind.txt", "r");
	for (int i = 0; i < 72756; i++) { fscanf(fseq, "%d", &seq[i]); } //id:1-72756
	fclose(fseq);	
}
void localization(int opt)
{
	img = cvCreateImage(cvSize(SIZE, SIZE), IPL_DEPTH_8U, 1);
	Mat z;
	z = Mat::zeros(SIZE, SIZE, CV_8UC1);
	IplImage* src = NULL;
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", depdir, seq[i] + 1, ".png");
		Mat savez = imread(filename, 0);
		resize(savez, z, Size(SIZE, SIZE));
		nowcnt++;
		//imshow("", z);
		//waitKey(1);
		for (int row = 0; row < SIZE; row++)
		{
			for (int col = 0; col < SIZE; col++)
			{
				data[nowcnt - 1][0][row][col] = z.at<uchar>(row, col) / 255.0;
			}
		}
	}
}
void genHDF5()
{

	hid_t fileid, datasetid, dataspaceid, labelid, labelspaceid, datasetoriid, dataspaceoriid, datasetwhiteid, dataspacewhiteid;
	herr_t status;
	fileid = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4], dimsdataori[2];
	hsize_t dimslabel[2];
	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = SIZE;
	dimsdata[3] = SIZE;

	dimslabel[0] = DIML1;
	dimslabel[1] = DIML2;

	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/depthdata", H5T_IEEE_F32LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/doflabel", H5T_IEEE_F32LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	H5Dwrite(datasetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Dwrite(labelid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);

	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);


}
void genorigindep()
{
	for (int i = 0; i < 72704 / DIML1; i++)
	{
		for (int j = 0; j < DIML1; j++)
		{
			for (int k = 0; k < 55; k++)
			{
				label[j][k] = all[i*DIML1 + j][k];
			}

		}
		st = i*DIML1;
		en = (i + 1)*DIML1 - 1;
		nowcnt = 0;
		sprintf(file, "%s%d%s", depprefix, i + 1, ".h5");
		localization(1);
		genHDF5();
	}
}
int main()
{
	init();
	genorigindep();
	return 0;
}

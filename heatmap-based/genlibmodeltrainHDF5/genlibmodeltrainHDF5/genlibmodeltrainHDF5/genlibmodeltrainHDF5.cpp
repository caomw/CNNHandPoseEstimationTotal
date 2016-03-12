// genlibmodeltrainHDF5.cpp : �������̨Ӧ�ó������ڵ㡣
//


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
#define SIZE 224
#define datasetdir "J:\\cnnhandtotal\\cnntraindata\\size85\\"
#define DIML1 6063
#define DIML2 93
#define fileprefix "K:\\lib"
using namespace std;
using namespace cv;
int st, en,nowcnt;
IplImage *img;
float data[DIML1][1][SIZE][SIZE];
float label[DIML1][DIML2];
float all[72756][DIML2];
char file[111];
int seq[111111];
void init()
{
	FILE *fin = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyz.txt", "r");
	FILE *fseq = fopen("D:\\CNN\\genlibmodeltrainHDF5\\seq.txt", "r");
	for (int i = 0; i < 72756; i++) { fscanf(fseq, "%d", &seq[i]);  } //id:1-72756
	fclose(fseq);
	for (int i = 0; i < 72756; i++)
	{
		for (int j = 0; j < 93; j++)
		{
			fscanf(fin, "%f", &all[i][j]);			
		}
			
	}
	fclose(fin);
}
void localization()
{	
	img = cvCreateImage(cvSize(SIZE, SIZE), IPL_DEPTH_8U, 1);
	Mat z;	
	z = Mat::zeros(SIZE, SIZE, CV_8UC1);	
	IplImage* src = NULL;
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, seq[i], ".png");		
		Mat savez = imread(filename, 0);
		nowcnt++;		
		resize(savez, z, Size(SIZE,SIZE));
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
	datasetid = H5Dcreate(fileid, "/data", H5T_IEEE_F32LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label", H5T_IEEE_F32LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	H5Dwrite(datasetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	
	H5Dwrite(labelid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);

	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	

	status = H5Dclose(labelid);

	
	status = H5Sclose(labelspaceid);

	status = H5Fclose(fileid);

}
int main()
{	
	init();
	for (int i = 0; i < 72756/DIML1; i++)
	{	
		for (int j = 0; j < DIML1; j++)
		{
			for (int k = 0; k < 93; k++)
			{
				label[j][k] = all[seq[i*DIML1 + j]-1][k];				
			}
			
		}
		st = i*DIML1;
		en = (i+1)*DIML1-1;
		nowcnt = 0;
		sprintf(file, "%s%d%s", fileprefix, i+1, ".h5");
		localization();
		genHDF5();
	}
	return 0;
}

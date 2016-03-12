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
#include<eigen/dense>
#define st 12127
#define en 24252
#define file96 "D:\\CNN\\handjoint\\newtrain\\trialnew96_11.h5"
#define file48 "D:\\CNN\\handjoint\\newtrain\\trialnew48_11.h5"
#define file24 "D:\\CNN\\handjoint\\newtrain\\trialnew24_11.h5"
#define DIML1 en-st+1
#define DIML2 6776
using namespace std;
using namespace cv;
using namespace Eigen;
int jx[14], jy[14], cnt = 0;
double rotx[14], roty[14];
double a[111][111], data96[en - st + 1][1][96][96], data48[en - st + 1][1][48][48], data24[en - st + 1][1][24][24], label[DIML1][DIML2];

void localization()
{	
	IplImage* dst96 = cvCreateImage(cvSize(96, 96), IPL_DEPTH_64F, 1);
	IplImage* dst48 = cvCreateImage(cvSize(48, 48), IPL_DEPTH_64F, 1);
	IplImage* dst24 = cvCreateImage(cvSize(24, 24), IPL_DEPTH_64F, 1);
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];		
		sprintf(filename, "%s%d%s", "F:\\cnnhandtotal\\AA96\\", i, ".png");		
		IplImage* src = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);   //ÔØÈë»Ò¶ÈÍ¼Ïñ£¬Éî¶ÈÎª8U
		cvConvertScale(src, dst96, 1.0);		
		for (int j = 0; j < 96; j++)
		{
			for (int k = 0; k < 96; k++)
			{
				data96[i - st][0][j][k] = ((double *)(dst96->imageData + dst96->widthStep*k))[j] / 255.0;				
			}
		}



		sprintf(filename, "%s%d%s", "F:\\cnnhandtotal\\AA48\\", i, ".png");
		src = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);   //ÔØÈë»Ò¶ÈÍ¼Ïñ£¬Éî¶ÈÎª8U
		cvConvertScale(src, dst48, 1.0);
		for (int j = 0; j < 48; j++)
		{
			for (int k = 0; k < 48; k++)
			{
				data48[i - st][0][j][k] = ((double *)(dst48->imageData + dst48->widthStep*k))[j] / 255.0;
			}
		}


		sprintf(filename, "%s%d%s", "F:\\cnnhandtotal\\AA24\\", i, ".png");
		src = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);   //ÔØÈë»Ò¶ÈÍ¼Ïñ£¬Éî¶ÈÎª8U
		cvConvertScale(src, dst24, 1.0);
		for (int j = 0; j < 24; j++)
		{
			for (int k = 0; k < 24; k++)
			{
				data24[i - st][0][j][k] = ((double *)(dst24->imageData + dst24->widthStep*k))[j] / 255.0;
			}
		}


		char probname[111];
		sprintf(probname, "%s%d%s", "F:\\cnnhandtotal\\probtrain\\", i, ".txt");
		FILE *finprob = fopen(probname, "r");
		for (int j = 0; j < 6776; j++)
		{
			fscanf(finprob, "%lf", &label[i - st][j]);
		}
		fclose(finprob);							
	}
}
void genHDF5()
{
	//96*96
	hid_t fileid, datasetid, dataspaceid, labelid, labelspaceid;
	herr_t status;
	fileid = H5Fcreate(file96, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4], dimslabel[2];
	dimsdata[0] = en - st + 1;
	dimsdata[1] = 1;
	dimsdata[2] = 96;
	dimsdata[3] = 96;
	dimslabel[0] = en - st + 1;
	dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data96", H5T_IEEE_F64LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label96", H5T_IEEE_F64LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96);
	H5Dwrite(labelid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);

	//48*48

	fileid = H5Fcreate(file48, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	dimsdata[0] = en - st + 1;
	dimsdata[1] = 1;
	dimsdata[2] = 48;
	dimsdata[3] = 48;
	dimslabel[0] = en - st + 1;
	dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data48", H5T_IEEE_F64LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label48", H5T_IEEE_F64LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data48);
	H5Dwrite(labelid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);

	//24*24

	fileid = H5Fcreate(file24, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	dimsdata[0] = en - st + 1;
	dimsdata[1] = 1;
	dimsdata[2] = 24;
	dimsdata[3] = 24;
	dimslabel[0] = en - st + 1;
	dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data24", H5T_IEEE_F64LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label24", H5T_IEEE_F64LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data24);
	H5Dwrite(labelid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(labelid);
	status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);
}
int main()
{
	localization();
	genHDF5();
	return 0;
}
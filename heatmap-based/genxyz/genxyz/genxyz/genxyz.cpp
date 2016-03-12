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
#define nyudir "G:\\nyu_hand_dataset_v2\\dataset\\train\\"
#define DIML1 6063
#define DIML2 93
#define SIZE 224
#define xyzprefix "F:\\libmodel\\onlyxyz\\"
using namespace std;
using namespace cv;
int st, en;
int minh, minw, maxh, maxw;
float gt[72758][93];
float data[DIML1][1][SIZE][SIZE];
float label[DIML1][DIML2];
int seq[72758];
char file[111];
void init()
{
	FILE *fseq = fopen("D:\\CNN\\genlibmodeltrainHDF5\\seq.txt", "r");
	for (int i = 0; i < 72756; i++) { fscanf(fseq, "%d", &seq[i]); } //id:1-72756
	fclose(fseq);
	FILE *fin = fopen("J:\\cnnhandtotal\\libhandmodelexp\\onlyxyz.txt", "r");
	for (int i = 0; i < 72756; i++)
	{
		for (int j = 0; j < 31; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				fscanf(fin, "%f", &gt[i][j*3+k]);
			}
		}
		for (int j = 0; j < 31; j++)
		{
			//cout << gt[i][j*3] << " " << gt[i][j*3+1] << " " << gt[i][j*3+2] << "\n";
			gt[i][j*3] = (gt[i][j*3] - gt[i][24*3]) / 200.0;
			gt[i][j*3+1] = (gt[i][j*3+1] - gt[i][24*3+1]) / 200.0;
			gt[i][j*3+2] = (min((float)1150.0,gt[i][j*3+2]) - 750) / 200.0;
			//cout << gt[i][j][0] << " " << gt[i][j][1] << " " << gt[i][j][2] << "\n";
		}
	}
	fclose(fin);
}

void genxyz()
{
	char filenameimg[111];
	char filenameskin[111];
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		sprintf(filenameimg, "%sdepth_%d_%07d.png", nyudir, 1, seq[i]);
		Mat img = imread(filenameimg);

		sprintf(filenameskin, "%ssynthdepth_%d_%07d.png", nyudir, 1, seq[i]);
		Mat skin = imread(filenameskin);

		minh = 1111;
		minw = 1111;
		maxh = 0;
		maxw = 0;
		for (int h = 1; h < 480; h++)
		{
			for (int w = 1; w < 640; w++)
			{
				if (!(skin.at<Vec3b>(h, w)[0] == 0 && skin.at<Vec3b>(h, w)[1] == 0))
				{
					//cout << h << " " << w << "\n";
					minh = min(minh, h);
					maxh = max(maxh, h);
					minw = min(minw, w);
					maxw = max(maxw, w);
				}
			}
		}
		//cout << minh << " " << maxh << " " << minw << " " << maxw << "\n";
		minh = max(0, minh - 10); maxh = min(480, maxh + 10);
		minw = max(0, minw - 10); maxw = min(640, maxw + 10);
		if (maxw - minw > maxh - minh)
		{
			int delrow = maxh - minh, delcol = maxw - minw;
			int t = min(minh, (delcol - delrow) / 2);
			if (maxh + (delcol - delrow) - t > 480)
				t = maxh - 480 + (delcol - delrow);
			minh = minh - t;
			maxh = maxh + (delcol - delrow) - t;
		}
		else
		{
			int delcol = maxw - minw, delrow = maxh - minh;
			int t = min(minw, (delrow - delcol) / 2);
			if (maxw + (delrow - delcol) - t > 640)
				t = maxw - 640 + (delrow - delcol);
			minw = minw - t;
			maxw = maxw + (delrow - delcol) - t;
		}
		int mind = 1111, maxd = 0;
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{
				if (skin.at<Vec3b>(h, w)[0] == 0 && skin.at<Vec3b>(h, w)[1] == 0 && skin.at<Vec3b>(h, w)[2] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				if (depth > 1900) continue;
				mind = min(mind, depth);
				maxd = max(maxd, depth);
			}
		}
		Mat middle = Mat::zeros(Size(maxw - minw + 1, maxh - minh + 1), CV_32FC1);
		Mat middle2 = Mat::zeros(Size(maxw - minw + 1, maxh - minh + 1), CV_32FC1);
		float min1 = 111.0, max1 = -111.0;
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{
				middle.at<float>(h - minh, w - minw) = 2.0;
				if (skin.at<Vec3b>(h, w)[0] == 0 && skin.at<Vec3b>(h, w)[1] == 0 && skin.at<Vec3b>(h, w)[2] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				//if (depth != 0) cout << h << " " << w << " " << depth << "\n";
				middle.at<float>(h - minh, w - minw) = (depth - 750.0) / 200.0;
				
				min1 = min(min1,(float) ((depth - 750.0) / 200.0));
				max1 = max(max1,(float) ((depth - 750.0) / 200.0));
			}
		}
		/*
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{
				middle2.at<float>(h - minh, w - minw) = 1.0;
				if (skin.at<Vec3b>(h, w)[0] == 0 && skin.at<Vec3b>(h, w)[1] == 0 && skin.at<Vec3b>(h, w)[2] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				middle2.at<float>(h - minh, w - minw) = ((depth - 750.0) / 200.0-min1)/(max1-min1);
		
			}
		}
		imshow("", middle);
		waitKey(0);
		imshow("", middle2);
		waitKey(0);*/		
		Mat resizemiddle = Mat::zeros(Size(maxw - minw + 1, maxh - minh + 1), CV_32FC1);
		resize(middle, resizemiddle, Size(224, 224));
		//imshow("", resizemiddle);
		//waitKey(1);
		/*
		for (int i = 0; i < 224; i++)
		{
			for (int j = 0; j < 224; j++)
				cout << resizemiddle.at<float>(i, j) << " ";
			cout << "\n";
		}
		imshow("", resizemiddle);
		waitKey(1);*/
		for (int h = 0; h < 224; h++)
		{
			for (int w = 0; w < 224; w++)
				data[i - st][0][h][w] = resizemiddle.at<float>(h, w);
		}
	}
}
void hdf5()
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
void genh5()
{	
	for (int i = 0; i < 72756 / DIML1; i++)
	{
		for (int j = 0; j < DIML1; j++)
		{
			for (int k = 0; k < 93; k++)
			{
				label[j][k] = gt[seq[i*DIML1 + j] - 1][k]; //seq start from 1
			}

		}
		st = i*DIML1;
		en = (i + 1)*DIML1 - 1;		
		sprintf(file, "%s%d%s", xyzprefix, i + 1, ".h5");		
		genxyz();
		hdf5();
	}
}
int main()
{
	init();
	genh5();
	return 0;
}
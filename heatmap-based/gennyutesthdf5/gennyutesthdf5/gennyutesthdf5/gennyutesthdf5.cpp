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
#define percent 85 //valid hand size percent
#define datasetdir "J:\\cnnhandtotal\\cnntraindata\\res96\\"
#define jointprefix "J:\\cnnhandtotal\\cnntraindata\\sizeall.txt"
#define trainprefix "J:\\cnnhandtotal\\cnntraindata\\local"
int st, en;
const int sizemode[3] = { 96, 48, 24 };
const int ll[6] = { 1, 363781, 145513, 509293, 291025, 654805 };
//#define file96prefix "G:\\skinanddepth\\datasmall160withuv85size96_"
#define file96prefix "G:\\nyutest8248"
#define N 72756

char file96[111];
#define DIML1 8248
#define DIML2 6776

using namespace std;
using namespace cv;
int nowcnt = 0;
float b[111][111];
float a[111][111], data96[DIML1][1][96][96];
float label[DIML1][DIML2];



Mat srcimg;
IplImage *img[3];
int all = 0;
int hsh[72758], sta[72758];
void init()
{
	int cnt = 0;
	for (int i = 2; i <= 8252 ; i++)
	{
		if (i >= 6936 && i <= 6938) continue;
		char heatmapfile[111];
		sprintf(heatmapfile, "%s%d%s", "J:\\cnnhandtotal\\probtest\\", i, ".txt");
		FILE *fh = fopen(heatmapfile, "r");
		for (int j = 0; j < 6776; j++)
		{
			fscanf(fh, "%f", &label[cnt][j]);
			if (label[cnt][j] < 1e-3) label[cnt][j] = 0.0;			
		}
		cnt++;
		fclose(fh);
	}
}
void localization()
{
	st = 1;
	en = 8248;
	for (int i = 0; i < 3; i++)
		img[i] = cvCreateImage(cvSize(96, 96), IPL_DEPTH_8U, 1);

	Mat z[3];
	Mat z2;
	for (int mode = 0; mode < 2; mode++)
	{
		z[mode] = Mat::zeros(sizemode[mode], sizemode[mode], CV_8UC1);
	}
	IplImage* src = NULL;
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char filename[111];
		sprintf(filename, "%s%d%s", datasetdir, i, ".png");
		//srcimg = imread(filename, 0);				
		Mat savez = imread(filename, 0);
		nowcnt++;
		all++;



		for (int mode = 0; mode < 1; mode++)
		{
			int size = sizemode[mode];

			resize(savez, z[mode], Size(size, size));


			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					((uchar *)(img[mode]->imageData + img[mode]->widthStep*row))[col] = z[mode].at<uchar>(row, col);
					//((uchar *)(img[1]->imageData + img[1]->widthStep*row))[col] = z[mode].at<uchar>(row, col);


					switch (mode)
					{
					case 0:data96[nowcnt - 1][0][row][col] = 1.0 - z[mode].at<uchar>(row, col) / 255.0;
						//data96ori[nowcnt - 1][row*96+col] = (z[mode].at<uchar>(row, col)==255?0.0:1.0); //skin
						//data96ori[nowcnt - 1][row * 96 + col+96*96] = 1.0-z[mode].at<uchar>(row, col)/255.0; //depth background black
						//if (z[mode].at<uchar>(row, col) == 255) data96ori[nowcnt - 1][0][row][col] = 0; else data96ori[nowcnt - 1][0][row][col] =1.0;
						//data96ori[nowcnt - 1][0][row][col] = 1.0 - double(z[mode].at<uchar>(row, col)) / 255.0;
						//data96white[nowcnt - 1][0][row][col] =  double(z[mode].at<uchar>(row, col)) / 255.0;
						break;
					}
				}
			}

			//waitKey(0);
			char savefile[111], savefile2[111];
			sprintf(savefile, "%s%d%s%d%s", trainprefix, sizemode[mode], "\\", all, ".png");
			//sprintf(savefile2, "%s%d%s%d%s", trainprefix, sizemode[1], "\\", all, ".png");
			//cvSaveImage(savefile, img[mode]);
			//cvSaveImage(savefile2, img[1]);
		}
	}

}
void genHDF5()
{
	//96*96
	hid_t fileid, datasetid, dataspaceid, labelid, labelspaceid, datasetoriid, dataspaceoriid, datasetwhiteid, dataspacewhiteid;
	herr_t status;
	fileid = H5Fcreate(file96, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4], dimsdataori[2];
	hsize_t dimslabel[2];
	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = 96;
	dimsdata[3] = 96;
	/*
	dimsdataori[0] = DIML1;
	dimsdataori[1] = 2*96*96;
	*/
	dimslabel[0] = DIML1;
	dimslabel[1] = DIML2;

	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data96", H5T_IEEE_F32LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	/*
	dataspaceoriid = H5Screate_simple(2, dimsdataori, NULL);
	datasetoriid = H5Dcreate(fileid, "/data96ori", H5T_IEEE_F32LE, dataspaceoriid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	*/
	labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	labelid = H5Dcreate(fileid, "/label96", H5T_IEEE_F32LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	H5Dwrite(datasetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96);
	//H5Dwrite(datasetoriid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96ori);
	H5Dwrite(labelid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);

	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	//status = H5Dclose(datasetoriid);

	status = H5Dclose(labelid);

	//status = H5Sclose(dataspaceoriid);
	status = H5Sclose(labelspaceid);

	status = H5Fclose(fileid);


}
int main()
{
	init();
	for (int id = 0; id < 1; id++)
	{
		for (int i = 1; i <= 1; i++)
		{
			//72756/DIML1
			st = ll[id] - 1 + (i - 1)*DIML1 + 1;
			en = ll[id] - 1 + i*DIML1;
			nowcnt = 0;
			sprintf(file96, "%s%d%s", file96prefix, id * 72756 / DIML1 + i, ".h5");
			localization();
			genHDF5();

		}
	}
	return 0;
}
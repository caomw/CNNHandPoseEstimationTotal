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
#define datasetdir "J:\\cnnhandtotal\\cnntraindata\\size85\\"
#define jointprefix "J:\\cnnhandtotal\\cnntraindata\\sizeall.txt"
#define trainprefix "J:\\cnnhandtotal\\cnntraindata\\local"
int st, en;
const int sizemode[3] = { 96, 48, 24 };
const int ll[6] = { 1, 363781, 145513, 509293, 291025, 654805 };
#define file96prefix "G:\\skin\\trial961_"
#define N 72756
//#define file48prefix "J:\\newtrain\\trial48_"
//#define file24prefix "J:\\newtrain\\trial24_"
char file96[111];//, file48[111], file24[111];
#define DIML1 12126
#define DIML2 6776
using namespace std;
using namespace cv;
int nowcnt = 0;
float b[111][111];
float a[111][111], data96[DIML1][1][96][96], data96ori[DIML1][1][96][96];//,data96white[DIML1][1][96][96];
float label[DIML1][DIML2];//, data48[DIML1][1][48][48], data24[DIML1][1][24][24];
float ox[N+4][15], oy[N+4][15];
Mat srcimg;
IplImage *img[3];
int all = 0;
int hsh[72758], sta[72758];
void init()
{
	FILE *fjoint = fopen(jointprefix, "r");
	for (int i = 0; i < N; i++)
	{		
		for (int j = 0; j < 14; j++) fscanf(fjoint, "%f %f", &ox[i][j], &oy[i][j]); 		

		//cout << "\n";
		
	}
	fclose(fjoint);
	for (int i = 1; i <= 72756; i++)
	{
		int t = (rand()*rand() + rand()) % 72756 + 1;
		while (hsh[t] == 1) t = (rand()*rand() + rand()) % 72756 + 1;
		hsh[t] = 1;
		sta[i] = i;
		//cout << i<<" "<<t << "\n ";
	}
}
void localization()
{
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
		sprintf(filename, "%s%d%s", datasetdir, sta[i], ".png");		
		//srcimg = imread(filename, 0);				
		Mat savez = imread(filename, 0);
		nowcnt++;
		all++;
		/*for (int k = 0; k < 14; k++)
		{
			for (int u = 1; u <= 22; u++)
			{
				for (int v = 1; v <= 22; v++)
				{
					float t = exp(-1.0 / (2 * 1 * 1)*(pow(u - ox[sta[all]-1][k] - 0.5, 2) + pow(v - oy[sta[all]-1][k] - 0.5, 2)));
					if (u == 1 && v == 1 && t>0.10)
					{
						cout << ox[all][k]<<" "<<oy[all][k]<<" "<<u << " " << v << " " << t << "\n";
						int p = 1;
					}
					
					if (t < 1e-6) t = 0.0;
					label[nowcnt - 1][k * 484 + (v-1) * 22 + u-1] = t;
				}
			}
		}*/
		

		for (int mode = 0; mode < 1; mode++)
		{
			int size = sizemode[mode];

			resize(savez, z[mode], Size(size, size));
			//Swap Row And Col
			/*for (int row = 0; row < size; row++)
			{
				for (int col = row; col < size; col++)
				{
					int t = z[mode].at<uchar>(row, col);
					z[mode].at<uchar>(row, col) = z[mode].at<uchar>(col, row);
					z[mode].at<uchar>(col, row) = t;
				}
			}*/
			//imshow("", z[mode]);
			//waitKey(0);
			
			float aveall = 0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					aveall += z[mode].at<uchar>(row, col);
				}
			}
			aveall /= (size*size);
			float stdall = 0.0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					stdall += pow(z[mode].at<uchar>(row, col) - aveall, 2);
				}
			}
			stdall /= (size*size);
			stdall = sqrt(stdall);
			z2 = z[mode];
			float pmin = 1111.0, pmax = 0.0;
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					float ave = 0;
					int cnt = 0;
					for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
					{
						for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
						{
							ave += z2.at<uchar>(k, l);
							cnt++;
						}
					}
					ave = ave / cnt;
					float stdnow = 0;
					for (int k = max(0, row - (4 - 1 - mode)); k <= min(size - 1, row + (4 - 1 - mode)); k++)
					{
						for (int l = max(0, col - (4 - 1 - mode)); l <= min(size - 1, col + (4 - 1 - mode)); l++)
						{
							stdnow += pow(z2.at<uchar>(k, l) - ave, 2);
						}
					}
					stdnow = stdnow / cnt;
					stdnow = sqrt(stdnow);
					a[row][col] = (z[mode].at<uchar>(row, col) - ave) / (stdnow + stdall);
					pmin = min(pmin, a[row][col]);
					pmax = max(pmax, a[row][col]);
				}
			}
			//imshow("", z[mode]);
			//waitKey(0);
			for (int row = 0; row < size; row++)
			{
				for (int col = 0; col < size; col++)
				{
					((uchar *)(img[mode]->imageData + img[mode]->widthStep*row))[col] = 255-z[mode].at<uchar>(row, col) ;
					((uchar *)(img[1]->imageData + img[1]->widthStep*row))[col] = z[mode].at<uchar>(row, col);
					
					
					switch (mode)
					{
					case 0:data96[nowcnt - 1][0][row][col] = 1.0-z[mode].at<uchar>(row, col) / 255.0;
						data96ori[nowcnt - 1][0][row][col] = z[mode].at<uchar>(row, col) / 255.0;
						//if (z[mode].at<uchar>(row, col) == 255) data96ori[nowcnt - 1][0][row][col] = 0; else data96ori[nowcnt - 1][0][row][col] =1.0;
						   //data96ori[nowcnt - 1][0][row][col] = 1.0 - double(z[mode].at<uchar>(row, col)) / 255.0;
						   //data96white[nowcnt - 1][0][row][col] =  double(z[mode].at<uchar>(row, col)) / 255.0;
						break;
					}
				}
			}
			
			//waitKey(0);
			char savefile[111],savefile2[111];
			sprintf(savefile, "%s%d%s%d%s", trainprefix, sizemode[mode], "\\", all, ".png");
			sprintf(savefile2, "%s%d%s%d%s", trainprefix, sizemode[1], "\\", all, ".png");
			cvSaveImage(savefile, img[mode]);
			cvSaveImage(savefile2, img[1]);
		}
	}
	
}
void genHDF5()
{
	//96*96
	hid_t fileid, datasetid, dataspaceid, labelid, labelspaceid ,datasetoriid,dataspaceoriid,datasetwhiteid,dataspacewhiteid;
	herr_t status;
	fileid = H5Fcreate(file96, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4], dimslabel[2],dimsdataori[4],dimsdatawhite[4];
	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = 96;
	dimsdata[3] = 96;

	dimsdataori[0] = DIML1;
	dimsdataori[1] = 1;
	dimsdataori[2] = 96;
	dimsdataori[3] = 96;

	/*dimsdatawhite[0] = DIML1;
	dimsdatawhite[1] = 1;
	dimsdatawhite[2] = 96;
	dimsdatawhite[3] = 96;*/

	//dimslabel[0] = DIML1;
	//dimslabel[1] = 6776;
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data96", H5T_IEEE_F32LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	dataspaceoriid = H5Screate_simple(4, dimsdataori, NULL);
	datasetoriid = H5Dcreate(fileid, "/data96ori", H5T_IEEE_F32LE, dataspaceoriid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	/*dataspacewhiteid = H5Screate_simple(4, dimsdatawhite, NULL);
	datasetwhiteid = H5Dcreate(fileid, "/data96white", H5T_IEEE_F32LE, dataspacewhiteid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);*/

	//labelspaceid = H5Screate_simple(2, dimslabel, NULL);
	//labelid = H5Dcreate(fileid, "/label96", H5T_IEEE_F32LE, labelspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(datasetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96);
	H5Dwrite(datasetoriid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96ori);
	//H5Dwrite(datasetwhiteid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data96white);
	//H5Dwrite(labelid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Dclose(datasetoriid);
	//status = H5Dclose(datasetwhiteid);
	status = H5Sclose(dataspaceoriid);
	//status = H5Dclose(labelid);
	//status = H5Sclose(labelspaceid);
	status = H5Fclose(fileid);


}
int main()
{
	init();
	for (int id = 0; id < 1; id++)
	{
		for (int i = 1; i <= 72756 / DIML1; i++)
		{
			st = ll[id]-1 +(i-1)*DIML1 +1 ; 
			en = ll[id]-1 + i*DIML1;
			nowcnt = 0;
			sprintf(file96, "%s%d%s", file96prefix, id*72756/DIML1+i, ".h5");
			localization();
			genHDF5();
		}
	}
	return 0;
}
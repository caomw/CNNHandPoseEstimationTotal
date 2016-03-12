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
#include<ctime>
#define datasetdir "G:\\nyu_hand_dataset_v2\\dataset\\test\\synthdepth_1_"
#define realdepdir "G:\\nyu_hand_dataset_v2\\dataset\\test\\depth_1_"
#define savedir "J:\\cnnhandtotal\\cubedata\\"
int st, en;
const int sizemode[3] = { 224, 48, 24 };
const int ll[6] = { 1, 363781, 145513, 509293, 291025, 654805 };
#define file224prefix "J:\\cnnhandtotal\\cubedata\\data224_"
#define N 72756

char file224[111];
#define DIML1 6063
using namespace std;
using namespace cv;
int nowcnt = 0;
float b[111][111];
float  data224[DIML1][1][224][224];
int hsh[72758], sta[72758];
float a[480][640];
float reala[480][640];
void init()
{
	/*for (int i = 1; i <= 72756; i++)
	{
		int t = (rand()*rand() + rand()) % 72756 + 1;
		while (hsh[t] == 1) t = (rand()*rand() + rand()) % 72756 + 1;
		hsh[t] = 1;
		sta[i] = t;

	}*/
	
}
void work()
{
	for (int i = st; i <= en; i++)
	{
		cout << i << "\n";
		char picname[111];
		sprintf(picname,"%s%07d%s", datasetdir, i, ".png");
		char picrealdep[111];
		sprintf(picrealdep, "%s%07d%s", realdepdir, i, ".png");
		Mat src = imread(picname);
		Mat realdep = imread(picrealdep);
		//imshow("", realdep);
		//waitKey(0);
		int sumu = 0, sumv = 0, sumd = 0,cnt=0;
		for (int u = 0; u < 480; u++)
		{
			for (int v = 0; v < 640; v++)
			{
				a[u][v] = src.at<Vec3b>(u, v)[1] * 255 + src.at<Vec3b>(u, v)[0];
				reala[u][v] = realdep.at<Vec3b>(u, v)[1] * 255 + realdep.at<Vec3b>(u, v)[0];
				if (a[u][v]>0.0)
				{
					sumu = sumu + u;
					sumv = sumv + v;
					sumd = sumd + a[u][v];
					cnt = cnt + 1;
				}
			}
		}		
		int comu = sumu / cnt, comv = sumv / cnt, comd = sumd / cnt;
		Mat res = Mat::zeros(Size(300, 300), CV_8UC1);
		for (int u = comu - 150; u < comu + 150; u++)
		{
			for (int v = comv - 150; v < comv + 150; v++)
			{
				
				if (u < 0 || u >= 480 || v < 0 || v >= 640 || a[u][v]<1)
				{
					res.at<uchar>(u - (comu - 150) ,v - (comv - 150)) = 255.0;
					
				}
				else
				{
					res.at<uchar>(u - (comu - 150), v - (comv - 150)) = ((0.5 - (reala[u][v] - comd) / 150.0)+1)/2.0*255.0;
						
					
				}
			}
		}
		
		resize(res, res, Size(224, 224));
		imshow("", res);
		waitKey(1);
		char savename[111];
		sprintf(savename, "%s%s%d%s", savedir, "imgtestdep\\", i, ".png");
		imwrite(savename, res);
		for (int u = 0; u < 224; u++)
		{
			for (int v = 0; v < 224; v++)
			{
				data224[i - st][0][u][v] = res.at<uchar>(u, v)/255.0;
			}
		}
		//imshow("", res);
		//waitKey(0);
	}
	
	
}
void genHDF5()
{
	//224*224
	hid_t fileid, datasetid, dataspaceid;
	herr_t status;
	fileid = H5Fcreate(file224, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsdata[4];
	dimsdata[0] = DIML1;
	dimsdata[1] = 1;
	dimsdata[2] = 224;
	dimsdata[3] = 224;

	
	dataspaceid = H5Screate_simple(4, dimsdata, NULL);
	datasetid = H5Dcreate(fileid, "/data224", H5T_IEEE_F32LE, dataspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	

	
	H5Dwrite(datasetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data224);
	
	
	status = H5Dclose(datasetid);
	status = H5Sclose(dataspaceid);
	status = H5Fclose(fileid);


}
int main()
{
	srand(time(0));
	init();
	for (int id = 0; id < 1; id++)
	{
		for (int i = 1; i <= 72756 / DIML1; i++)
		{
			st = ll[id]-1 +(i-1)*DIML1 +1 ; 
			en = ll[id]-1 + i*DIML1;
			nowcnt = 0;
			sprintf(file224, "%s%d%s", file224prefix, id*72756/DIML1+i, ".h5");
			work();
			//genHDF5();
		}
	}
	return 0;
}
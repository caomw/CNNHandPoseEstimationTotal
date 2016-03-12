#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<iostream>
#include<cmath>
#include<ctime>
#define nyudir "G:\\nyu_hand_dataset_v2\\dataset\\test\\"
#define MAXN 111111
#define BALL 32
#define st 2
#define en 8252
using namespace std;
using namespace cv;
int minh, minw, maxh, maxw;
float ave0=754;
float predu[MAXN][BALL], predv[MAXN][BALL], gtu[MAXN][BALL], gtv[MAXN][BALL], scale[MAXN];
float err = 0.0;
float nowpixel = 0.0;
float convert(float x, float scale, float portion)
{
	return (x - 2.24 / 2.0) / (2.24 / 2.0)*1.0 *1/ (1.0)*1.0 ;
}
int get(Mat img, float u, float v)
{
	return img.at<Vec3b>(v, u)[1] * 256 + img.at<Vec3b>(v, u)[0];
}
int main()
{
	srand(time(0));
	FILE *fin = fopen("D:\\CNN\\calctesterror\\preduvgqp360000.txt", "r");
	FILE *fout = fopen("D:\\CNN\\calctesterror\\testerrorgqp360000.txt", "w");
	FILE *fdep = fopen("D:\\CNN\\calctesterror\\depth.txt", "w");
	FILE *fgt = fopen("J:\\cnnhandtotal\\libhandmodelexp\\gttest.txt", "r");
	char filenameimg[111];
	char filenameskin[111];
	
	for (int i = st; i <= en; i++)
	{
		cout << i << " ";
		for (int j = 0; j < 31; j++)
		{
			fscanf(fin, "%f%f%f%f%f", &predu[i][j], &predv[i][j], &gtu[i][j], &gtv[i][j], &scale[i]);			
			
		}
		sprintf(filenameimg, "%sdepth_%d_%07d.png", nyudir, 1, i);
		Mat img = imread(filenameimg);

		sprintf(filenameskin, "%ssynthdepth_%d_%07d.png", nyudir, 1, i);
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
					minh = min(minh, h);
					maxh = max(maxh, h);
					minw = min(minw, w);
					maxw = max(maxw, w);
				}
			}
		}
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

		int sum = 0, cnt = 0;
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{

				if (skin.at<Vec3b>(h, w)[0] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				sum += depth;
				cnt++;
			}
		}
		float ave = sum / cnt;
		int mind = 1111, maxd = 0;
		for (int h = minh; h <= maxh; h++)
		{
			for (int w = minw; w <= maxw; w++)
			{

				if (skin.at<Vec3b>(h, w)[0] == 0) continue;
				int depth = img.at<Vec3b>(h, w)[1] * 255 + img.at<Vec3b>(h, w)[0];
				if (depth > 1900) continue;
				mind = min(mind, depth);
				maxd = max(maxd, depth);

			}
		}

		float avedepth = sum / cnt;
		fprintf(fdep, "%d %d %.2f", mind, maxd, avedepth);
		float portion = ave0 / max(ave0, ave);
		//float portion = 1.0;
		cout << portion <<" "<<scale[i]<<" "<<avedepth<<" ";
		float centeru = (minw + maxw) / 2.0;
		float centerv = (minh + maxh) / 2.0;
		float size = (maxh - minh); //=maxw-minw
		float nowerr = 0.0;
		int cntnum = 0;		
		//portion = 1.0;
		for (int j = 0; j < 31; j++)
		{
			if (j == 0 || j == 3 || j == 5 || j == 8 || j == 10 || j == 13 || j == 15 || j == 18 || j == 24 || j == 25 || j == 26 || j == 28 || j == 29 || j == 30)
			{
				if (i == 5 && j == 0)
				{
					int t = rand() % 10;
				}
				float pu = convert(predu[i][j], scale[i], portion);
				float pv = convert(predv[i][j], scale[i], portion);
				float gu = convert(gtu[i][j], scale[i], portion);
				float gv = convert(gtv[i][j], scale[i], portion);
				
				pu = pu*size / 2.0 + centeru;
				pv = pv*size / 2.0 + centerv;
				
				gu = gu*size / 2.0 + centeru;
				gv = gv*size / 2.0 + centerv;
				circle(skin, Point(pu, pv), 3, Scalar(255, 255, 255), -2);
				
				int tpu = (int)pu;
				int tpv = (int)pv;
				float pd=maxd;
				pd = min(pd, (float)get(img, tpu, tpv));
				pd = min(pd, (float)get(img, tpu, tpv+1));
				pd = min(pd, (float)get(img, tpu+1, tpv));
				pd = min(pd, (float)get(img, tpu+1, tpv + 1));
				

				//float pd = (tpu + 1 - pu)*(tpv + 1 - pv)*get(img, tpu, tpv) + (pu - tpu)*(tpv + 1 - pv)*get(img, tpu + 1, tpv) + (tpu + 1 - pu)*(pv - tpv)*get(img, tpu, tpv + 1) + (pu - tpu)*(pv - tpv)*get(img, tpu + 1, tpv + 1);
				int tgu = (int)gu;
				int tgv = (int)gv;
				float gd = maxd;
				gd = min(gd, (float)get(img, tgu, tgv));
				gd = min(gd, (float)get(img, tgu, tgv + 1));
				gd = min(gd, (float)get(img, tgu + 1, tgv));
				gd = min(gd, (float)get(img, tgu + 1, tgv + 1));
				//float gd = (tgu + 1 - gu)*(tgv + 1 - gv)*get(img, tgu, tgv) + (gu - tgu)*(tgv + 1 - gv)*get(img, tgu + 1, tgv) + (tgu + 1 - gu)*(gv - tgv)*get(img, tgu, tgv + 1) + (gu - tgu)*(gv - tgv)*get(img, tgu + 1, tgv + 1);
				float px = (pu / 640.0 - 0.5)*pd;
				float py = (0.5 - pv / 480.0)*pd;
				float a, b, c;
				fscanf(fgt, "%f %f %f", &a, &b, &c);
				nowpixel += sqrt(pow(pu - gu, 2) + (pv - gv, 2));
				if (gd >1000)
				{
					gd = c;
				}
				float gx = (gu / 640.0 - 0.5)*gd;
				float gy = (0.5 - gv / 480.0)*gd;
				float err;
				int flag = 1;
										
				if (pd>1000 )
				{
					flag = 2;
					cntnum++;
					if (abs(pd-gd)>80)
					{
						px = (pu / 640.0 - 0.5)*avedepth;
						py = (0.5 - pv / 480.0)*avedepth;
					}
					if (abs(pd-gd)>80)
					{
						gx = (pu / 640.0 - 0.5)*avedepth;
						gy = (0.5 - pv / 480.0)*avedepth;
					}
					px = (pu / 640.0 - 0.5)*avedepth;
					py = (0.5 - pv / 480.0)*avedepth;
					err = sqrt(pow(px - gx, 2) + pow(py - gy, 2) );
					nowerr += err;

				}
				else
				{
					flag = 2;
					cntnum++;
					if (abs(pd - gd) > 80)
					{
						px = (pu / 640.0 - 0.5)*avedepth;
						py = (0.5 - pv / 480.0)*avedepth;

						gx = (gu / 640.0 - 0.5)*avedepth;
						gy = (0.5 - gv / 480.0)*avedepth;

						err = sqrt(pow(px - gx, 2) + pow(py - gy, 2));
						nowerr += err;

					}
					else
					{
						err = sqrt(pow(px - gx, 2) + pow(py - gy, 2) + pow(pd - gd, 2));
						nowerr += err;
					}
					
				}				
				fprintf(fout, "%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n ", px,gx,py,gy,pd,gd, int(flag==2)*err);
				
			}
		}
		imshow("", skin);
		waitKey(1);
		if (cntnum != 0)
		{
			nowerr /= cntnum;
			err += nowerr;
		}
		
		fprintf(fout, "%.4f %d\n",nowerr, cntnum);
		cout << nowerr << "\n";
	}
	
	fprintf(fout, "%.4f\n", err);
	err /= (en - st + 1);
	fprintf(fout, "%.4f\n", err);
	nowpixel /= ((en - st + 1) * 14);
	cout << "pixel error:" << nowpixel<<"\n";
	fclose(fin);
	fclose(fout);
	fclose(fdep);
	fclose(fgt);
	return 0;
}
// calcerror.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<algorithm>
#include<cmath>
#define st 0
#define en 1000
using namespace std;
float standard[72767][100], pred[100];
int seq[111111];
int main()
{
	FILE *fstandard = fopen("J:\\cnnhandtotal\\libhandmodelexp\\xyz.txt","r");
	FILE *fpred = fopen("J:\\cnnhandtotal\\libhandmodelexp\\predictuv.txt", "r");
	FILE *fseq = fopen("D:\\CNN\\genlibmodeltrainHDF5\\seq.txt", "r");
	for (int i = 0; i < 72756; i++)
	{
		for (int j = 0; j < 93; j++)
		{
			fscanf(fstandard, "%f", &standard[i][j]);
			standard[i][j] *= 100.0;
		}
	}
	float ans = 0.0;
	for (int i = 0; i < 72756; i++) { fscanf(fseq, "%d", &seq[i]); seq[i]--; };
	for (int i = st; i < en; i++)
	{
		int id = seq[i];
		for (int j = 0; j < 93; j++)
		{
			fscanf(fpred, "%f",&pred[j]);
		}
		float sum = 0.0;
		for (int j = 0; j < 31; j++)
		{
			//j == 0 || j == 3 || j == 5 || j == 8 || j == 10 || j == 13 || j == 15 || j == 18 || j == 25 || j == 26 || j == 28 || j == 29 || j == 30 || j==24
			if ( j== 24)
			{
				float x = pred[j * 3], y = pred[j * 3 + 1];
				float xerr = (x - standard[id][j * 3]) / 640 * 800.0, yerr = (y - standard[id][j * 3 + 1]) / 480 * 800.0;
				float err = sqrt(xerr*xerr + yerr*yerr);
				cout << x << " " << standard[id][j * 3] << "\n";
				sum += err;
			}
			
		}
		sum /= 1.0;
		ans += sum;
	}
	ans /= (en - st + 1);
	printf("%.4f\n", ans);
	fclose(fseq);
	fclose(fpred);
	fclose(fstandard);
	return 0;
}
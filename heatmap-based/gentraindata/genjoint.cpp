#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<iostream>
using namespace std;
int main()
{
	FILE *fin=fopen("J:\\cnnhandtotal\\cnntraindata\\ori.txt","r");
	FILE *fout=fopen("J:\\cnnhandtotal\\cnntraindata\\sizeall.txt","w");
	for (int i=1;i<=72756;i++)
	{
		cout<<"size 100 no   flip"<<i<<"\n";
		for (int j=1;j<=28;j++)
		{
			double x;
			fscanf(fin,"%lf",&x);
			fprintf(fout,"%.4f ", 11+(x-11.0)*1.00);
		}
		fprintf(fout,"\n");
	}
	fclose(fin);
	fin=fopen("J:\\cnnhandtotal\\cnntraindata\\ori.txt","r");
	for (int i=1;i<=72756;i++)
	{
		cout<<"size 100 with flip"<<i<<"\n";
		for (int j=1;j<=14;j++)
		{
			double x,y;
			fscanf(fin,"%lf %lf",&x,&y);
			fprintf(fout,"%.4f %.4f ", 22.0-(11+(x-11.0)*1.00),11+(y-11.0)*1.00);
		}
		fprintf(fout,"\n");
	}
	fclose(fin);
	fin=fopen("J:\\cnnhandtotal\\cnntraindata\\ori.txt","r");
	for (int i=1;i<=72756;i++)
	{
		cout<<"size  90 no   flip"<<i<<"\n";
		for (int j=1;j<=28;j++)
		{
			double x;
			fscanf(fin,"%lf",&x);
			fprintf(fout,"%.4f ", 11+(x-11.0)*0.90);
		}
		fprintf(fout,"\n");
	}
	fclose(fin);
	fin=fopen("J:\\cnnhandtotal\\cnntraindata\\ori.txt","r");
	for (int i=1;i<=72756;i++)
	{
		cout<<"size  90 with flip"<<i<<"\n";
		for (int j=1;j<=14;j++)
		{
			double x,y;
			fscanf(fin,"%lf %lf",&x,&y);
			fprintf(fout,"%.4f %.4f ", 22.0-(11+(x-11.0)*0.90),11+(y-11.0)*0.90);
		}
		fprintf(fout,"\n");
	}
	fclose(fin);
	fin=fopen("J:\\cnnhandtotal\\cnntraindata\\ori.txt","r");
	for (int i=1;i<=72756;i++)
	{
		cout<<"size  80 no   flip"<<i<<"\n";
		for (int j=1;j<=28;j++)
		{
			double x;
			fscanf(fin,"%lf",&x);
			fprintf(fout,"%.4f ", 11+(x-11.0)*0.80);
		}
		fprintf(fout,"\n");
	}
	fclose(fin);
	fin=fopen("J:\\cnnhandtotal\\cnntraindata\\ori.txt","r");
	for (int i=1;i<=72756;i++)
	{
		cout<<"size  80 with flip"<<i<<"\n";
		for (int j=1;j<=14;j++)
		{
			double x,y;
			fscanf(fin,"%lf %lf",&x,&y);
			fprintf(fout,"%.4f %.4f ", 22.0-(11+(x-11.0)*0.80),11+(y-11.0)*0.80);
		}
		fprintf(fout,"\n");
	}
	fclose(fin);
	fclose(fout);
	return 0;
}

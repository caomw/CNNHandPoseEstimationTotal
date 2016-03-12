#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<iostream>
#define N 8248
using namespace std;
double ori[N][14][2],target[N][14][2],lstm[N+1000][14][2];
double lstmans[14],orians[14];
int main()
{
	FILE *fori=fopen("D:\\CNN\\lstmheatmapalljoint\\lstmheatmapalljoint\\lstmheatmapalljoint\\oriout.txt","r");
	for (int i=0;i<N;i++)
	{
		for (int j=0;j<14;j++)
		{
		    fscanf(fori,"%lf %lf",&ori[i][j][0],&ori[i][j][1]);
		    ori[i][j][0]-=1.0; ori[i][j][1]-=1.0;
		}
	}
	fclose(fori);
	FILE *flstm=fopen("D:\\CNN\\lstmheatmapalljoint\\lstmheatmapalljoint\\lstmheatmapalljoint\\lstmout.txt","r");
	for (int i=0;i<N;i++) 
	{
		for (int j=0;j<14;j++)
		{
			fscanf(flstm,"%lf %lf",&lstm[i][j][0],&lstm[i][j][1]);
		}
	}	
	
	fclose(flstm);
	FILE *ftarget=fopen("D:\\CNN\\lstmheatmapalljoint\\lstmheatmapalljoint\\lstmheatmapalljoint\\targetout.txt","r");
	for (int i=0;i<N;i++) 
	{
	   for (int j=0;j<14;j++)
	   {	   	
	   		fscanf(ftarget,"%lf %lf",&target[i][j][0],&target[i][j][1]);		   
	   }
	}
	fclose(ftarget);
	
	for (int i=0;i<N;i++)
	{
		for (int j=0;j<14;j++)
		{
			orians[j]+=sqrt(pow(target[i][j][0]-ori[i][j][0],2)+pow(target[i][j][1]-ori[i][j][1],2));
			lstmans[j]+=sqrt(pow(target[i][j][0]-lstm[i][j][0],2)+pow(target[i][j][1]-lstm[i][j][1],2));		
		}		
	}
	double totori=0.0,totlstm=0.0;
	for (int i=0;i<14;i++)
	{		
		orians[i]/=N;
		lstmans[i]/=N;
		totori+=orians[i];
		totlstm+=lstmans[i];
		printf("# %d ----ori: %.6f  ",i,orians[i]/22.0*96.0); 
		printf("lstm: %.6f\n",lstmans[i]/22.0*96.0);
	}	
	totori/=14.0; totlstm/=14.0;
	
	printf("Total ori: %.6f ; total lstm: %.6f\n",totori/22.0*96.0,totlstm/22.0*96.0);
	return 0;
}

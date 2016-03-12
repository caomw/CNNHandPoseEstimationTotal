#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<iostream>
#define N 6000
using namespace std;
double ori[N][2],target[N][2],lstm[N+1000][2];
double lstmans,orians;
int main()
{
	FILE *fori=fopen("D:\\CNN\\lstmheatmap\\oriout.txt","r");
	for (int i=0;i<N;i++)
	{
	    fscanf(fori,"%lf %lf",&ori[i][0],&ori[i][1]);
	    
	}
	fclose(fori);
	FILE *flstm=fopen("D:\\CNN\\lstmheatmap\\lstmout.txt","r");
	for (int i=0;i<N;i++) fscanf(flstm,"%lf %lf",&lstm[i][0],&lstm[i][1]);
	for (int i=0;i<N;i++)
	{
		if (lstm[i][0]==0.0)
		{
		   cout<<i<<"\n";
		   break;
		}
	}
	fclose(flstm);
	FILE *ftarget=fopen("D:\\CNN\\lstmheatmap\\targetout.txt","r");
	for (int i=0;i<N;i++) 
	{
	   fscanf(ftarget,"%lf %lf",&target[i][0],&target[i][1]);		   
	}
	fclose(ftarget);
	lstmans=0.0;
	orians=0.0;
	for (int i=0;i<N;i++)
	{
		
		orians+=sqrt(pow(target[i][0]-ori[i][0],2)+pow(target[i][1]-ori[i][1],2));
		lstmans+=sqrt(pow(target[i][0]-lstm[i][0],2)+pow(target[i][1]-lstm[i][1],2));		
		if (i>=4800 && i<=4840)
		{
			//cout<<target[i][0]<<" "<<lstm[i][0]<<" "<<target[i][1]<<" "<<lstm[i][1]<<"\n";
//			cout<<sqrt(pow(target[i][0]-ori[i][0],2)+pow(target[i][1]-ori[i][1],2))<<" "<<sqrt(pow(target[i][0]-lstm[i][0],2)+pow(target[i][1]-lstm[i][1],2))<<"\n";
		}
	}
	orians/=N;
	lstmans/=N;
	printf("ori: %.6f\n",orians);
	printf("lstm: %.6f\n",lstmans);
	return 0;
}

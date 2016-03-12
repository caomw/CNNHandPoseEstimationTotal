#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<iostream>
#define N 8248
using namespace std;
double ori[N][2],target[N][2],lstm[N+1000][2];
double lstmans,orians;
int main()
{
	FILE *fori=fopen("D:\\CNN\\lstmheatmapjoint10\\lstmheatmapjoint10\\lstmheatmapjoint10\\oriout.txt","r");
	for (int i=0;i<N;i++)
	{
	    fscanf(fori,"%lf %lf",&ori[i][0],&ori[i][1]);
	    ori[i][0]=ori[i][0]-1.0;
	    ori[i][1]=ori[i][1]-1.0;
	}
	fclose(fori);
	FILE *flstm=fopen("D:\\CNN\\lstmheatmapjoint10\\lstmheatmapjoint10\\lstmheatmapjoint10\\lstmout.txt","r");
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
	FILE *ftarget=fopen("D:\\CNN\\lstmheatmapjoint10\\lstmheatmapjoint10\\lstmheatmapjoint10\\targetout.txt","r");
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

	}
	orians/=N;
	lstmans/=N;
	printf("ori: %.6f\n",orians/22.0*96.0);
	printf("lstm: %.6f\n",lstmans/22.0*96.0);
	return 0;
}

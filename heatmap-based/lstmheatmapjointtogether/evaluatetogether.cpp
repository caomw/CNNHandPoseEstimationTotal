#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<iostream>
#define N 8248
using namespace std;
double ori[N][2],target[N][2],lstm[N+1000][2];
double lstmans,orians,totori,totlstm;
int main()
{
	for (int joint=0;joint<14;joint++)
	{
		//if (joint==8 || joint==10) continue;
		char oriname[111];
		sprintf(oriname,"%s%d%s","D:\\CNN\\lstmheatmapjointtogether\\lstmheatmapjointtogether\\lstmheatmapjointtogether\\ori",joint,"out.txt");
		FILE *fori=fopen(oriname,"r");
		for (int i=0;i<N;i++) 
		{
		   fscanf(fori,"%lf %lf",&ori[i][0],&ori[i][1]);		    
		   ori[i][0]-=(joint==8 || joint==10);
		   ori[i][1]-=(joint==8 || joint==10);
		}
		fclose(fori);
		
		char lstmname[111];
		sprintf(lstmname,"%s%d%s","D:\\CNN\\lstmheatmapjointtogether\\lstmheatmapjointtogether\\lstmheatmapjointtogether\\lstm",joint,"out.txt");
		FILE *flstm=fopen(lstmname,"r");
		for (int i=0;i<N;i++) fscanf(fori,"%lf %lf",&lstm[i][0],&lstm[i][1]);		    		
		fclose(flstm);
		
		char targetname[111];
		sprintf(targetname,"%s%d%s","D:\\CNN\\lstmheatmapjointtogether\\lstmheatmapjointtogether\\lstmheatmapjointtogether\\target",joint,"out.txt");
		FILE *ftarget=fopen(targetname,"r");
		for (int i=0;i<N;i++) fscanf(fori,"%lf %lf",&target[i][0],&target[i][1]);		    		
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
		totori+=orians;
		totlstm+=lstmans;
		printf("%d ori: %.6f ",joint,orians/22.0*96.0);
		printf("   lstm: %.6f\n",lstmans/22.0*96.0);
	}	
	totori/=14.0;
	totlstm/=14.0;
	printf("Total ori: %.6f ; total lstm: %.6f\n",totori/22.0*96.0,totlstm/22.0*96.0);
	return 0;
}

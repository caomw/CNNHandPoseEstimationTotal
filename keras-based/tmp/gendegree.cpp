#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<algorithm>
#include<cmath>
#define N 100
#define eps 1e-6
using namespace std;
double x[48][3];
double a[96][96];
double ans,maxx;
int main()
{
	FILE *fin=fopen("D:\\CNN\\handmodel\\randdeg.txt","r");
	ans=0.0;
	for (int id=0;id<N;id++)
	{
		cout<<id<<"\n";
		for (int i=0;i<48;i++)
		{		
			fscanf(fin,"%lf %lf %lf",&x[i][0],&x[i][1],&x[i][2]);
		}
		//cout<<x[30][0]<<" "<< x[30][1]<<"\n";
		memset(a,0,sizeof(a));
		for (int i=0;i<48;i++)
		{
			for (int j=0;j<96;j++)
			{
				for (int k=0;k<96;k++)
				{
					a[j][k]=a[j][k]+x[i][2]*exp((pow((x[i][0]/x[i][2]+0.5)*96 - k, 2) + pow((0.5+(-x[i][1])/( x[i][2]))*96-j, 2)) *(-1.0 / 50));
				}
			}
		}
		double maxx=0.0;
		for (int i=0;i<96;i++)
		{
			for (int j=0;j<96;j++)
			{
				if (a[i][j]-maxx>eps) maxx=a[i][j];
			}
		}
		ans=ans+maxx;
	}
	ans=ans/N;
	printf("%.4f\n",ans);
	fclose(fin);
	return 0;
}

#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<iostream>
#include<cmath>
#include<ctime>
#define eps 1e-10
#define MAXN 11111111 
#define NUM1 1
#define NUM2 1
#define NUM3 1
using namespace std;
int seq1[MAXN],seq2[MAXN],seq3[MAXN],ind[MAXN],seq[MAXN],number1,number2,number3,cnt;
int a[MAXN],hsh[MAXN];
int main()
{
	FILE *fin=fopen("rankerrxyz.txt","r");
	srand(time(0));
	for (int i=0;i<72704;i++)
	{
		int id;
		float t;
		fscanf(fin,"%d %f",&id,&t);
		if (t-25>eps)  seq1[number1++]=id;
		//else if (t-20>eps) seq2[number2++]=id;
		//else if (t-16.5>eps) seq3[number3++]=id;
	}
	for (int i=0;i<NUM1;i++) 
	{
		for (int j=0;j<number1;j++)
		{
			seq[cnt++]=seq1[j];
		}
	}
	/*
	for (int i=0;i<NUM2;i++) 
	{
		for (int j=0;j<number2;j++)
		{
			seq[cnt++]=seq2[j];
		}
	}
	for (int i=0;i<NUM3;i++) 
	{
		for (int j=0;j<number3;j++)
		{
			seq[cnt++]=seq3[j];
		}
	}*/
	for (int i=1;i<=72756;i++) seq[cnt++]=i;
	FILE *fout=fopen("seqxyz4.txt","w");
	for (int i=0;i<cnt;i++) 
	{
		int tmp=(rand()*rand()+rand())%cnt; //0~cnt-1
		while (hsh[tmp]==1) tmp=(rand()*rand()+rand())%cnt;
		hsh[tmp]=1;
		a[i]=seq[tmp];
		fprintf(fout,"%d\n",a[i]);
	}
	cout<<number1<<" "<<number2<<" "<<number3<<" "<<cnt<<"\n";
	fclose(fin);
	fclose(fout);
	return 0;
}

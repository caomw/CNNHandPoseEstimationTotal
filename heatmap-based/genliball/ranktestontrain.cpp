#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<iostream>
using namespace std;

struct node
{
	int id;
	float key;
	node(){}
	node(int id,float key):id(id),key(key){}
}err[111111];
bool cmp(node u,node v)
{
	return u.key>v.key;
}
int main()
{
	FILE *fin=fopen("testontraindata.txt","r");
	FILE *finind=fopen("D:\\CNN\\gendeeppriordata\\ind.txt","r"); //0-72756
	for (int i=0;i<72704;i++)
	{
		fscanf(finind,"%d",&err[i].id); err[i].id++; //1-72757
		fscanf(fin,"%f",&err[i].key);
	}
	sort(err,err+72704,cmp);
	FILE *fout=fopen("rankerrxyz.txt","w");
	for (int i=0;i<72704;i++)
	{
		fprintf(fout,"%d %.4f\n",err[i].id,err[i].key);
	}
	fclose(fin);
	fclose(finind);
	fclose(fout);
	return 0;
}

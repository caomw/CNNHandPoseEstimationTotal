st=16; en=251;
%oripicdir=sprintf('%s','F:\\cnnhandtotal\\rdfcnn\\pic2\\');
oripicdir=sprintf('%s','D:\\CNN\\0728\\CNN1\\0728\\res96\\');
NUM=10;
MINLIGHT=0.50;
for i=st:en
   %{ 
    if ~((i>=0 && i<=94) || (i>=112 && i<=165) || (i>=190 && i<=218) || (i>=232 && i<=251)) 
        %continue;
    end
   %}
    fprintf('%d\n',i);    
    err=0.0;
    count=0;
    fitname=sprintf('%s%d%s','D:\\CNN\\fit',i,'.txt');
    foutfit=fopen(fitname,'wt');
    %%%%Allocate space
    peakx0=zeros(2,14,NUM); peaky0=zeros(2,14,NUM); peaksse=zeros(2,14,NUM); peakrmse=zeros(2,14,NUM); peakrsquare=zeros(2,14,NUM); peakadjrsquare=zeros(2,14,NUM);
    choose=zeros(14);
    value=zeros(14,NUM);
    for j=0:13
        %if ~(j==0 || j==3 || j==4 || j==5 || j==7 || j==9 || j==11 || j==13 || j==6 || j==8 || j==10 || j==12) continue; end;
        %if ~ (j==6 || j==8 || j==10 || j==3 || j==4 || j==5) continue; end;
        file=sprintf('%s%d_%d%s','D:\\CNN\\joint\\',i,j,'.txt');
        %fout=fopen(file,'wt');
        %filename1=sprintf('%s%d_%d%s','D:\\CNN\\results\\r_',i,j,'.png');        
        filename1=sprintf('%s%d_%d%s','D:\\CNN\\CNN1\\results\\r_',i,j,'.png');
        z=imread(filename1);                
        if numel(z)==1452
           z=rgb2gray(z);
        end;
        z=mat2gray(z);
        %%%Preprocessing
        savez=z; savez1=z;                 
        maxrow=zeros(NUM); maxcol=zeros(NUM); 
        %Find the top NUM peaks
        for num=1:NUM
            [M,I]=max(savez1(:));
            value(j+1,num)=M;
            [maxrow(num),maxcol(num)]=ind2sub(size(z),I);
            for irow=max(1,maxrow(num)-1): min(3-(maxrow(num)-max(1,maxrow(num)-1)+1)+maxrow(num),22)
                for icol=max(1,maxcol(num)-1):min(3-(maxcol(num)-max(1,maxcol(num)-1)+1)+maxcol(num),22)
                    savez1(irow,icol)=0.0;
                end
            end             
        end
        
        %%%%%%%%%%%%%%%%
        for T=2:2            
            for num=1:NUM
                row1=max(1,maxrow(num)-T); row2=min(2*T+1-(maxrow(num)-max(1,maxrow(num)-T)+1)+maxrow(num),22);
                col1=max(1,maxcol(num)-T); col2=min(2*T+1-(maxcol(num)-max(1,maxcol(num)-T)+1)+maxcol(num),22);                    
                savez1=savez;
                %%% restore saved z
                for irow=1:22
                    for icol=1:22
                        if ~(irow>=row1 && irow<=row2 && icol>=col1 && icol<=col2)
                            savez1(irow,icol)=0.0;
                        end
                    end
                end        
                %%%Fit only (2*T+1)*(2*T+1) small area
                [x,y]=meshgrid(1:22,1:22);
                x=x(:);
                y=y(:);
                z=savez1(:);                       
                sx=double(maxcol(num)); sy=double(maxrow(num)); su=1.0;               
                [peakresult,peakfit]=createFit(x,y,z,sx,sy,su);                    
                peakx0(T,j+1,num)=peakresult.x0; peaky0(T,j+1,num)=peakresult.y0;     
                peaksse(T,j+1,num)=peakfit.sse; peakrmse(T,j+1,num)=peakfit.rmse; peakrsquare(T,j+1,num)=peakfit.rsquare; peakadjrsquare(T,j+1,num)=peakfit.adjrsquare;
            end  
        end
        
        fprintf(foutfit,'%d\n',j);
        for num=1:NUM
            %fprintf(foutfit,'    %d.%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n',num,value(num),peakx0(1,num),peaky0(1,num),peaksse(1,num),peakrmse(1,num),peakrsquare(1,num),peakadjrsquare(1,num));
            fprintf(foutfit,'    %d.%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n',num,value(j+1,num),peakx0(2,j+1,num),peaky0(2,j+1,num),peaksse(2,j+1,num),peakrmse(2,j+1,num),peakrsquare(2,j+1,num),peakadjrsquare(2,j+1,num));
        end                           
        fminrsquare=11111.0; fminadjrsquare=11111.0; maxnum=0;
        for k=1:NUM            
            if ((1.0-peakrsquare(2,j+1,k))<fminrsquare)+((1.0-peakadjrsquare(2,j+1,k))<fminadjrsquare)>=2 && (value(j+1,k)>MINLIGHT)
                maxnum=k;
                fminrsquare=1.0-peakrsquare(2,j+1,k); fminadjrsquare=1.0-peakadjrsquare(2,j+1,k);
            end
        end
        choose(j+1)=maxnum; %% Default choice
    end        
    fclose(foutfit);               
    fprintf('---');
    tchoose=choose;
    oripic=sprintf('%s%d%s',oripicdir,i+1,'.png');
    pic=imread(oripic);    
    %pic=rgb2gray(pic);
    pic=imresize(pic,[96 96]);
    %%Modify Finger 3 4 5
    choose=modifythumb(choose,value,peakrsquare,pic,peakx0,peaky0,0,0,NUM,0);    
    %%%Modify Finger 7 9 11 13
    choose=modifymiddle(choose,value,peakrsquare,pic,peakx0,peaky0,0,0,NUM);         
    %%%Modidy Finger 6 8 10 12
    choose=modifyup(choose,value,peakrsquare,pic,peakx0,peaky0,0,0,NUM);

     
     %%%%%%%%%%%%%%%%Finger 3 4 5 should be at the left of palm
    thumbleftarea=0; thumbrightarea=0;
     if peakx0(2,6+1,choose(6+1))>peakx0(2,8+1,choose(8+1)) && peakx0(2,8+1,choose(8+1))>peakx0(2,10+1,choose(10+1)) && peakx0(2,10+1,choose(10+1))>peakx0(2,12+1,choose(12+1)) && ...
        peakx0(2,7+1,choose(7+1))>peakx0(2,9+1,choose(9+1)) && peakx0(2,9+1,choose(9+1))>peakx0(2,11+1,choose(11+1)) && peakx0(2,11+1,choose(11+1))>peakx0(2,13+1,choose(13+1)) && ...
        peakx0(2,3+1,choose(3+1))>peakx0(2,0+1,choose(0+1)) && peakx0(2,4+1,choose(4+1))>peakx0(2,0+1,choose(0+1)) && peakx0(2,5+1,choose(5+1))>peakx0(2,0+1,choose(0+1))
        thumbleftarea=1;       
     end
     %%%%%%%%%%%%%%%%Finger 3 4 5 should be at the right of palm
    if peakx0(2,6+1,choose(6+1))<peakx0(2,8+1,choose(8+1)) && peakx0(2,8+1,choose(8+1))<peakx0(2,10+1,choose(10+1)) && peakx0(2,10+1,choose(10+1))<peakx0(2,12+1,choose(12+1)) && ...
        peakx0(2,7+1,choose(7+1))<peakx0(2,9+1,choose(9+1)) && peakx0(2,9+1,choose(9+1))<peakx0(2,11+1,choose(11+1)) && peakx0(2,11+1,choose(11+1))<peakx0(2,13+1,choose(13+1)) && ...
        peakx0(2,3+1,choose(3+1))<peakx0(2,0+1,choose(0+1)) && peakx0(2,4+1,choose(4+1))<peakx0(2,0+1,choose(0+1)) && peakx0(2,5+1,choose(5+1))<peakx0(2,0+1,choose(0+1))
        thumbrightarea=1;        
        
    end     
    if thumbleftarea==0 && thumbrightarea==0
        if peakx0(2,3+1,choose(3+1))<peakx0(2,0+1,choose(0+1)) && peakx0(2,4+1,choose(4+1))<peakx0(2,0+1,choose(0+1)) && peakx0(2,5+1,choose(5+1))<peakx0(2,0+1,choose(0+1)) 
            thumbleftarea=1;
        else
            if peakx0(2,3+1,choose(3+1))>peakx0(2,0+1,choose(0+1)) && peakx0(2,4+1,choose(4+1))>peakx0(2,0+1,choose(0+1)) && peakx0(2,5+1,choose(5+1))>peakx0(2,0+1,choose(0+1)) 
                thumbrightarea=1;
            end
        end
    end
    %%%Modify thumb middle up
     if thumbleftarea~=0 || thumbrightarea~=0
         choose=modifythumb(choose,value,peakrsquare,pic,peakx0,peaky0,thumbleftarea,thumbrightarea,NUM,0);
         choose=modifymiddle(choose,value,peakrsquare,pic,peakx0,peaky0,thumbleftarea,thumbrightarea,NUM);
         choose=modifyup(choose,value,peakrsquare,pic,peakx0,peaky0,thumbleftarea,thumbrightarea,NUM);
     end
     tooclose=0;
    for j=6:13
        if computedis2(peakx0(2,j+1,choose(j+1)),peaky0(2,j+1,choose(j+1)),peakx0(2,3+1,choose(3+1)),peaky0(2,3+1,choose(3+1)))<5.0
            tooclose=1;
            break;
        end
    end
    %%%Finger 3 4 5 should be modified;
    if tooclose==1
        choose=modifythumb(choose,value,peakrsquare,pic,peakx0,peaky0,thumbleftarea,thumbrightarea,NUM,1);
    end
    %%Check sucess or not
    tooclose=0;
    for j=6:13
        if computedis2(peakx0(2,j+1,choose(j+1)),peaky0(2,j+1,choose(j+1)),peakx0(2,3+1,choose(3+1)),peaky0(2,3+1,choose(3+1)))<5.0
            tooclose=1;
            break;
        end
    end
    %%%Modify 3 4 5 failure
    if tooclose==1
        choose(3+1)=tchoose(3+1); choose(4+1)=tchoose(4+1); choose(5+1)=tchoose(5+1);
    end
     %choose=tchoose;
    for j=0:13
        %if ~(j==0 || j==3 || j==4 || j==5 || j==7 || j==9 || j==11 || j==13 || j==6 || j==8 || j==10 || j==12) continue; end;
        file=sprintf('%s%d_%d%s','D:\\CNN\\joint\\',i,j,'.txt');
        fout=fopen(file,'wt');
        fprintf(fout,'%.6f %.6f\n',peakx0(2,j+1,choose(j+1)),peaky0(2,j+1,choose(j+1)));
        fclose(fout);
    end
end
function retchoose=modifythumb(choose,value,peakrsquare,pic,peakx0,peaky0,thumbleftarea,thumbrightarea,NUM,tooclose)
	%%%%%%%%%%%%%%%%Finger 3 4 5 Begin 
    CONFIDENCE=5;
    minfarthumb=zeros(CONFIDENCE); maxclosethumb=zeros(CONFIDENCE);
    flag=zeros(CONFIDENCE); max3=zeros(CONFIDENCE); max4=zeros(CONFIDENCE); max5=zeros(CONFIDENCE);
    for j=1:CONFIDENCE
        minfarthumb(j)=1111; maxclosethumb(j)=0;
        flag(j)=0;
    end    
    
     THUMBSQUARE=0.70; THUMBMINLIGHT=0.30;
     for j=1:NUM %% Finger 3 
        if peakrsquare(2,3+1,j)<0 ...
             %|| (peakrsquare(2,3+1,j)>THUMBSQUARE && value(3+1,j)<THUMBMINLIGHT)
            continue; 
        end        
        if computedis2(peakx0(2,3+1,j),peaky0(2,3+1,j),peakx0(2,1+1,choose(1+1)),peaky0(2,1+1,choose(1+1)))<14.0 || computedis2(peakx0(2,3+1,j),peaky0(2,3+1,j),peakx0(2,2+1,choose(2+1)),peaky0(2,2+1,choose(2+1)))<14.0
            continue;
        end        
        for k=1:NUM
            if peakrsquare(2,4+1,k)<0 ...
                %|| (peakrsquare(2,4+1,k)>THUMBSQUARE && value(4+1,k)<THUMBMINLIGHT)
                continue; 
            end
            if computedis2(peakx0(2,4+1,k),peaky0(2,4+1,k),peakx0(2,1+1,choose(1+1)),peaky0(2,1+1,choose(1+1)))<14.0 || computedis2(peakx0(2,4+1,k),peaky0(2,4+1,k),peakx0(2,2+1,choose(2+1)),peaky0(2,2+1,choose(2+1)))<14.0
                continue;
            end
            for l=1:NUM   
                if peakrsquare(2,5+1,l)<0 ...
                     %|| (peakrsquare(2,5+1,l)>THUMBSQUARE && value(5+1,l)<THUMBMINLIGHT)
                    continue; 
                end
                if tooclose==1
                    mark3=0;
                    for u=6:13
                        if computedis2(peakx0(2,u+1,choose(u+1)),peaky0(2,u+1,choose(u+1)),peakx0(2,3+1,j),peaky0(2,3+1,j))<12.0
                            mark3=1;
                            break;
                        end
                    end
                    mark4=0;
                    for u=6:13
                        if computedis2(peakx0(2,u+1,choose(u+1)),peaky0(2,u+1,choose(u+1)),peakx0(2,4+1,k),peaky0(2,4+1,k))<12.0
                            mark4=1;
                            break;
                        end
                    end
                    mark5=0;
                    for u=6:13
                        if computedis2(peakx0(2,u+1,choose(u+1)),peaky0(2,u+1,choose(u+1)),peakx0(2,5+1,l),peaky0(2,5+1,l))<12.0
                            mark5=1;
                            break;
                        end
                    end            
                    if mark3+mark4+mark5>=2                        
                        continue;
                    end           
                end
                if computedis2(peakx0(2,5+1,l),peaky0(2,5+1,l),peakx0(2,1+1,choose(1+1)),peaky0(2,1+1,choose(1+1)))<14.0 || computedis2(peakx0(2,5+1,l),peaky0(2,5+1,l),peakx0(2,2+1,choose(2+1)),peaky0(2,2+1,choose(2+1)))<14.0
                    continue;
                end
                if (checkempty(pic,max(1,int32(peaky0(2,3+1,j)/22.0*96.0)-5),min(96,int32(peaky0(2,3+1,j)/22.0*96.0)+5),max(1,int32(peakx0(2,3+1,j)/22.0*96.0)-5),min(96,int32(peakx0(2,3+1,j)/22.0*96.0)+5))<=20)+(checkempty(pic,max(1,int32(peaky0(2,4+1,k)/22.0*96.0)-5),min(96,int32(peaky0(2,4+1,k)/22.0*96.0)+5),max(1,int32(peakx0(2,4+1,k)/22.0*96.0)-5),min(96,int32(peakx0(2,4+1,k)/22.0*96.0)+5))<=20)+(checkempty(pic,max(1,int32(peaky0(2,5+1,l)/22.0*96.0)-5),min(96,int32(peaky0(2,5+1,l)/22.0*96.0)+5),max(1,int32(peakx0(2,5+1,l)/22.0*96.0)-5),min(96,int32(peakx0(2,5+1,l)/22.0*96.0)+5))<=20)>=2
                    continue;
                end%%Not too empty
                closethumb=min(computedis2(peakx0(2,3+1,j),peaky0(2,3+1,j),peakx0(2,4+1,k),peaky0(2,4+1,k)),(computedis2(peakx0(2,4+1,k),peaky0(2,4+1,k),peakx0(2,5+1,l),peaky0(2,5+1,l))));
                %max(min)
                farthumb=max(computedis2(peakx0(2,3+1,j),peaky0(2,3+1,j),peakx0(2,4+1,k),peaky0(2,4+1,k)),(computedis2(peakx0(2,4+1,k),peaky0(2,4+1,k),peakx0(2,5+1,l),peaky0(2,5+1,l))));
                %min(max)
                if (thumbleftarea==1 && (peakx0(2,3+1,j)<peakx0(2,0+1,choose(0+1)))+(peakx0(2,4+1,k)<peakx0(2,0+1,choose(0+1)))+(peakx0(2,5+1,l)<peakx0(2,0+1,choose(0+1)))==3)  || (thumbrightarea==1 && (peakx0(2,3+1,j)>peakx0(2,0+1,choose(0+1)))+(peakx0(2,4+1,k)>peakx0(2,0+1,choose(0+1)))+(peakx0(2,5+1,l)>peakx0(2,0+1,choose(0+1)))==3) ...
                || (thumbleftarea==0 && thumbrightarea==0 && (((peakx0(2,3+1,j)<peakx0(2,0+1,choose(0+1)))+(peakx0(2,4+1,k)<peakx0(2,0+1,choose(0+1)))+(peakx0(2,5+1,l)<peakx0(2,0+1,choose(0+1)))==3) ||  (peakx0(2,3+1,j)>peakx0(2,0+1,choose(0+1)))+(peakx0(2,4+1,k)>peakx0(2,0+1,choose(0+1)))+(peakx0(2,5+1,l)>peakx0(2,0+1,choose(0+1)))==3))
                    if (peakx0(2,4+1,k)>peakx0(2,3+1,j) && peakx0(2,5+1,l)>peakx0(2,4+1,k)) || (peakx0(2,4+1,k)<peakx0(2,3+1,j) && peakx0(2,5+1,l)<peakx0(2,4+1,k))
                        
                        for confidence=1:CONFIDENCE %90 85 80 75 70 65 60 55 50
                            if ((peakrsquare(2,3+1,j)>=1.00-0.05*confidence)+(peakrsquare(2,4+1,k)>=1.00-0.05*confidence)+(peakrsquare(2,5+1,l)>=1.00-0.05*confidence)>=3)                             
                                if closethumb<=14.0 && farthumb<=14.0 && farthumb<minfarthumb(confidence) || (farthumb==minfarthumb(confidence) && closethumb>maxclosethumb(confidence))
                                    flag(confidence)=1;
                                    minfarthumb(confidence)=min(minfarthumb(confidence),farthumb);
                                    maxclosethumb(confidence)=max(maxclosethumb(confidence),closethumb);                                
                                    max3(confidence)=j; max4(confidence)=k; max5(confidence)=l;                                    
                                end
                            end                                            
                        end                                                
                    end                                        
                end                
            end
        end
     end   
     retchoose=choose;
     for j=1:CONFIDENCE
         if flag(j)~=0
             retchoose(3+1)=max3(j); retchoose(4+1)=max4(j); retchoose(5+1)=max5(j);             
             break
         end
     end     
     %%%%%%%%%%%%%%%%Finger 3 4 5 End 
end
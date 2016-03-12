function retchoose=modifymiddle(choose,value,peakrsquare,pic,peakx0,peaky0,thumbleftarea,thumbrightarea,NUM)
    %%%%%%%%%%%%%%%%Finger 7 9 11 13  Begin        
    CONFIDENCE=5;
    minfarmiddle=zeros(CONFIDENCE); maxclosemiddle=zeros(CONFIDENCE);
    flag=zeros(CONFIDENCE); max7=zeros(CONFIDENCE); max9=zeros(CONFIDENCE); max11=zeros(CONFIDENCE); max13=zeros(CONFIDENCE);
    for j=1:CONFIDENCE
        minfarmiddle(j)=1111; maxclosemiddle(j)=0;
        flag(j)=0;
    end    
     
    MIDDLESQUARE=0.80; MIDDLEMINLIGHT=0.30;    
    
      for j=1:NUM %% Finger 7
        if peakrsquare(2,7+1,j)<0 ...
            || (peakrsquare(2,7+1,j)>MIDDLESQUARE && value(7+1,j)<MIDDLEMINLIGHT) 
            continue; 
        end        
        for k=1:NUM
            if peakrsquare(2,9+1,k)<0  ...
                || (peakrsquare(2,9+1,k)>MIDDLESQUARE && value(9+1,k)<MIDDLEMINLIGHT)  
                continue; 
            end
            for l=1:NUM
                if peakrsquare(2,11+1,l)<0  ...
                    || (peakrsquare(2,11+1,l)>MIDDLESQUARE && value(11+1,l)<MIDDLEMINLIGHT) 
                    continue; 
                end
                for m=1:NUM
                    if peakrsquare(2,13+1,m)<0 ...
                         || (peakrsquare(2,13+1,m)>MIDDLESQUARE && value(13+1,m)<MIDDLEMINLIGHT) 
                        continue;
                    end            
                    if (checkempty(pic,max(1,int32(peaky0(2,7+1,j)/22.0*96.0)-5),min(96,int32(peaky0(2,7+1,j)/22.0*96.0)+5),max(1,int32(peakx0(2,7+1,j)/22.0*96.0)-5),min(96,int32(peakx0(2,7+1,j)/22.0*96.0)+5))<10)+( checkempty(pic,max(1,int32(peaky0(2,9+1,k)/22.0*96.0)-5),min(96,int32(peaky0(2,9+1,k)/22.0*96.0)+5),max(1,int32(peakx0(2,9+1,k)/22.0*96.0)-5),min(96,int32(peakx0(2,9+1,k)/22.0*96.0)+5))<10)+(checkempty(pic,max(1,int32(peaky0(2,11+1,l)/22.0*96.0)-5),min(96,int32(peaky0(2,11+1,l)/22.0*96.0)+5),max(1,int32(peakx0(2,11+1,l)/22.0*96.0)-5),min(96,int32(peakx0(2,11+1,l)/22.0*96.0)+5))<10)+(checkempty(pic,max(1,int32(peaky0(2,13+1,m)/22.0*96.0)-5),min(96,int32(peaky0(2,13+1,m)/22.0*96.0)+5),max(1,int32(peakx0(2,13+1,m)/22.0*96.0)-5),min(96,int32(peakx0(2,13+1,m)/22.0*96.0)+5))<10)>=2
                        continue;
                    end
                    closemiddle=min(computedis2(peakx0(2,7+1,j),peaky0(2,7+1,j),peakx0(2,9+1,k),peaky0(2,9+1,k)),min(computedis2(peakx0(2,9+1,k),peaky0(2,9+1,k),peakx0(2,11+1,l),peaky0(2,11+1,l)),computedis2(peakx0(2,11+1,l),peaky0(2,11+1,l),peakx0(2,13+1,m),peaky0(2,13+1,m))));
                    closemiddleabs=min(computeabs(peakx0(2,7+1,j),peaky0(2,7+1,j),peakx0(2,9+1,k),peaky0(2,9+1,k)),min(computeabs(peakx0(2,9+1,k),peaky0(2,9+1,k),peakx0(2,11+1,l),peaky0(2,11+1,l)),computeabs(peakx0(2,11+1,l),peaky0(2,11+1,l),peakx0(2,13+1,m),peaky0(2,13+1,m))));
                    %max(min)
                    farmiddle=max(computedis2(peakx0(2,7+1,j),peaky0(2,7+1,j),peakx0(2,9+1,k),peaky0(2,9+1,k)),max(computedis2(peakx0(2,9+1,k),peaky0(2,9+1,k),peakx0(2,11+1,l),peaky0(2,11+1,l)),computedis2(peakx0(2,11+1,l),peaky0(2,11+1,l),peakx0(2,13+1,m),peaky0(2,13+1,m))));
                    farmiddleabs=max(computeabs(peakx0(2,7+1,j),peaky0(2,7+1,j),peakx0(2,9+1,k),peaky0(2,9+1,k)),max(computeabs(peakx0(2,9+1,k),peaky0(2,9+1,k),peakx0(2,11+1,l),peaky0(2,11+1,l)),computeabs(peakx0(2,11+1,l),peaky0(2,11+1,l),peakx0(2,13+1,m),peaky0(2,13+1,m))));
                    %min(max)          
                    if thumbleftarea==1
                        if ~(peakx0(2,13+1,m)<peakx0(2,11+1,l) && peakx0(2,11+1,l)<peakx0(2,9+1,k) && peakx0(2,9+1,k)<peakx0(2,7+1,j)) 
                            continue; 
                        end
                    end
                    if thumbrightarea==1
                        if ~(peakx0(2,13+1,m)>peakx0(2,11+1,l) && peakx0(2,11+1,l)>peakx0(2,9+1,k) && peakx0(2,9+1,k)>peakx0(2,7+1,j)) 
                            continue; 
                        end                        
                    end
                    if (peakx0(2,9+1,k)<peakx0(2,7+1,j) && peakx0(2,11+1,l)<peakx0(2,9+1,k) && peakx0(2,13+1,m)<peakx0(2,11+1,l)) || (peakx0(2,9+1,k)>peakx0(2,7+1,j) && peakx0(2,11+1,l)>peakx0(2,9+1,k) && peakx0(2,13+1,m)>peakx0(2,11+1,l)) 
                        for confidence=1:CONFIDENCE %90 85 80 75 70 65 60 55 50                            
                            if ((peakrsquare(2,7+1,j)>=0.90-0.05*confidence)+(peakrsquare(2,9+1,k)>=0.90-0.05*confidence)+(peakrsquare(2,11+1,l)>=0.90-0.05*confidence)+(peakrsquare(2,13+1,m)>=0.90-0.05*confidence)>=4)
                            
                               if closemiddleabs>=7.0 &&  farmiddleabs<=18.0 && farmiddle<24.0 && (closemiddle>maxclosemiddle(confidence) || farmiddle<minfarmiddle(confidence))
                                   %farmiddle<minfarmiddle(confidence) || (farmiddle==minfarmiddle(confidence) &&
                                   
                                  flag(confidence)=1;
                                  minfarmiddle(confidence)=min(minfarmiddle(confidence),farmiddle);
                                  maxclosemiddle(confidence)=max(maxclosemiddle(confidence),closemiddle);                                
                                  max7(confidence)=j; max9(confidence)=k; max11(confidence)=l;  max13(confidence)=m;                                   
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
             retchoose(7+1)=max7(j); retchoose(9+1)=max9(j); retchoose(11+1)=max11(j); retchoose(13+1)=max13(j);             
             break
         end
     end     
end    
     %%%%%%%%%%%%%%%%Finger 7 9 11 13 End
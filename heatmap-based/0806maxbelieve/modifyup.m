function retchoose=modifyup(choose,value,peakrsquare,pic,peakx0,peaky0,thumbleftarea,thumbrightarea,NUM)
     %%%%%%%%%%%%%%%%Finger 6 8 10 12  Begin        
    CONFIDENCE=5;
    minfarup=zeros(CONFIDENCE); maxcloseup=zeros(CONFIDENCE);
    flag=zeros(CONFIDENCE); max6=zeros(CONFIDENCE); max8=zeros(CONFIDENCE); max10=zeros(CONFIDENCE); max12=zeros(CONFIDENCE);    
    believe=zeros(CONFIDENCE,4);
    for j=1:CONFIDENCE
        minfarup(j)=1111; maxcloseup(j)=0;
        flag(j)=0;
    end    
     
    UPSQUARE=0.80; UPMINLIGHT=0.30;    
    
      for j=1:NUM %% Finger 6
        if peakrsquare(2,6+1,j)<0 ...
            || (peakrsquare(2,6+1,j)>UPSQUARE && value(6+1,j)<UPMINLIGHT)
            continue; 
        end        
        for k=1:NUM
            if peakrsquare(2,8+1,k)<0 ...
                || (peakrsquare(2,8+1,k)>UPSQUARE && value(8+1,k)<UPMINLIGHT)
                continue; 
            end
            for l=1:NUM
                if peakrsquare(2,10+1,l)<0 ...
                    || (peakrsquare(2,10+1,l)>UPSQUARE && value(10+1,l)<UPMINLIGHT) 
                    continue; 
                end
                for m=1:NUM
                    if (checkempty(pic,max(1,int32(peaky0(2,6+1,j)/22.0*96.0)-5),min(96,int32(peaky0(2,6+1,j)/22.0*96.0)+5),max(1,int32(peakx0(2,6+1,j)/22.0*96.0)-5),min(96,int32(peakx0(2,6+1,j)/22.0*96.0)+5))<10)+( checkempty(pic,max(1,int32(peaky0(2,8+1,k)/22.0*96.0)-5),min(96,int32(peaky0(2,8+1,k)/22.0*96.0)+5),max(1,int32(peakx0(2,8+1,k)/22.0*96.0)-5),min(96,int32(peakx0(2,8+1,k)/22.0*96.0)+5))<10)+(checkempty(pic,max(1,int32(peaky0(2,10+1,l)/22.0*96.0)-5),min(96,int32(peaky0(2,10+1,l)/22.0*96.0)+5),max(1,int32(peakx0(2,10+1,l)/22.0*96.0)-5),min(96,int32(peakx0(2,10+1,l)/22.0*96.0)+5))<10)+(checkempty(pic,max(1,int32(peaky0(2,12+1,m)/22.0*96.0)-5),min(96,int32(peaky0(2,12+1,m)/22.0*96.0)+5),max(1,int32(peakx0(2,12+1,m)/22.0*96.0)-5),min(96,int32(peakx0(2,12+1,m)/22.0*96.0)+5))<10)>=2
                        continue;
                    end
                    if peakrsquare(2,12+1,m)<0    ...
                        || (peakrsquare(2,12+1,m)>UPSQUARE && value(12+1,m)<UPMINLIGHT)
                        continue;
                    end
                    
                    closeup=min(computedis2(peakx0(2,6+1,j),peaky0(2,6+1,j),peakx0(2,8+1,k),peaky0(2,8+1,k)),min(computedis2(peakx0(2,8+1,k),peaky0(2,8+1,k),peakx0(2,10+1,l),peaky0(2,10+1,l)),computedis2(peakx0(2,10+1,l),peaky0(2,10+1,l),peakx0(2,12+1,m),peaky0(2,12+1,m))));
                    closeupabs=min(computeabs(peakx0(2,6+1,j),peaky0(2,6+1,j),peakx0(2,8+1,k),peaky0(2,8+1,k)),min(computeabs(peakx0(2,8+1,k),peaky0(2,8+1,k),peakx0(2,10+1,l),peaky0(2,10+1,l)),computeabs(peakx0(2,10+1,l),peaky0(2,10+1,l),peakx0(2,12+1,m),peaky0(2,12+1,m))));
                    %max(min)
                    farup=max(computedis2(peakx0(2,6+1,j),peaky0(2,6+1,j),peakx0(2,8+1,k),peaky0(2,8+1,k)),max(computedis2(peakx0(2,8+1,k),peaky0(2,8+1,k),peakx0(2,10+1,l),peaky0(2,10+1,l)),computedis2(peakx0(2,10+1,l),peaky0(2,10+1,l),peakx0(2,12+1,m),peaky0(2,12+1,m))));
                    farupabs=max(computeabs(peakx0(2,6+1,j),peaky0(2,6+1,j),peakx0(2,8+1,k),peaky0(2,8+1,k)),max(computeabs(peakx0(2,8+1,k),peaky0(2,8+1,k),peakx0(2,10+1,l),peaky0(2,10+1,l)),computeabs(peakx0(2,10+1,l),peaky0(2,10+1,l),peakx0(2,12+1,m),peaky0(2,12+1,m))));                    
                    
                    %min(max)                        
                     if thumbleftarea==1
                        if ~(peakx0(2,12+1,m)<peakx0(2,10+1,l) && peakx0(2,10+1,l)<peakx0(2,8+1,k) && peakx0(2,8+1,k)<peakx0(2,6+1,j)) 
                            continue; 
                        end
                    end
                    if thumbrightarea==1
                        if ~(peakx0(2,12+1,m)>peakx0(2,10+1,l) && peakx0(2,10+1,l)>peakx0(2,8+1,k) && peakx0(2,8+1,k)>peakx0(2,6+1,j)) 
                            continue; 
                        end                        
                    end
                    if (peakx0(2,8+1,k)>peakx0(2,6+1,j) && peakx0(2,10+1,l)>peakx0(2,8+1,k) && peakx0(2,12+1,m)>peakx0(2,10+1,l)) || (peakx0(2,8+1,k)<peakx0(2,6+1,j) && peakx0(2,10+1,l)<peakx0(2,8+1,k) && peakx0(2,12+1,m)<peakx0(2,10+1,l))                    
                        for confidence=1:CONFIDENCE %90 85 80 75 70 65 60 55 50                            
%                              if confidence==2 && j==2 && k==3 &&  m==1
%                                fprintf('l: %d closeup: %.6f maxcloseup: %.6f farup: %.6f minfarup: %.6f Now6: %.6f Now8: %.6f Now10: %.6f Now12: %.6f Best6: %.6f Best8: %.6f Best10: %.6f Best12: %.6f\n',l,closeup,maxcloseup(confidence),farup,minfarup(confidence),peakrsquare(2,6+1,j),peakrsquare(2,8+1,k),peakrsquare(2,10+1,l),peakrsquare(2,12+1,m),believe(confidence,1),believe(confidence,2),believe(confidence,3),believe(confidence,4));
%                              end
                            if ((peakrsquare(2,6+1,j)>=1.00-0.05*confidence)+(peakrsquare(2,8+1,k)>=1.00-0.05*confidence)+(peakrsquare(2,10+1,l)>=1.00-0.05*confidence)+(peakrsquare(2,12+1,m)>=1.00-0.05*confidence)>=4)
                               if closeupabs>=8.0 && farup<=32.0 && farupabs<=32.0 && (maxcloseup(confidence)==0 || (abs(closeup-maxcloseup(confidence))<=9.0 && abs(farup-minfarup(confidence))<=8.0 && ...
                                   (peakrsquare(2,6+1,j)>believe(confidence,1) || peakrsquare(2,8+1,k)>believe(confidence,2) || peakrsquare(2,10+1,l)>believe(confidence,3) ||  peakrsquare(2,12+1,m)>believe(confidence,4))))
                                   %closeup>maxcloseup(confidence) 
                                   %&& min(peakrsquare(2,6+1,j),min(peakrsquare(2,8+1,k),min(peakrsquare(2,10+1,l),peakrsquare(2,12+1,m))))>minrsquare(confidence)
                                   %
                               %if closeup>=8.0 && farup<18.0 && farup<minfarup(confidence) || (farup==minfarup(confidence) && closeup>maxcloseup(confidence))
                                  flag(confidence)=1;                                  
                                  minfarup(confidence)=min(minfarup(confidence),farup);
                                  maxcloseup(confidence)=max(maxcloseup(confidence),closeup);                                
                                  believe(confidence,1)=max(believe(confidence,1),peakrsquare(2,6+1,j));
                                  believe(confidence,2)=max(believe(confidence,2),peakrsquare(2,8+1,k));
                                  believe(confidence,3)=max(believe(confidence,3),peakrsquare(2,10+1,l));
                                  believe(confidence,4)=max(believe(confidence,4),peakrsquare(2,12+1,m));
                                 
                                  max6(confidence)=j; max8(confidence)=k; max10(confidence)=l;  max12(confidence)=m;                                   
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
             retchoose(6+1)=max6(j); retchoose(8+1)=max8(j); retchoose(10+1)=max10(j); retchoose(12+1)=max12(j);             
             break
         end
     end     
     %%%%%%%%%%%%%%%%Finger 6 8 10 12 End
end
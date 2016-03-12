function [x]=model(d)
    fp=fopen('initialHandParameter.txt','r');
    for i=1:60
        for j=1:3
            std(i,j)=fscanf(fp,'%lf',1);
            std(i,j)=std(i,j)/100.0;
        end        
    end
    fclose(fp);
   
    opalm=d(1:3)';
    rpalm=rotate(0,d(4))*rotate(1,d(5))*rotate(2,d(6));
    for i=1:35
        x(i,:)=rpalm*(std(i,:)')+opalm;
    end
    
    
    
    othumb=x(6,:)'; % rotation joint is sphere 1
    rthumb=rotate(1,d(7))*rotate(2,d(8));
    for i=36:37
       x(i,:)=rpalm*rthumb*(std(i,:)'-std(6,:)')+othumb;
    end

    othumbsecond=x(37,:)';
    rthumbsecond=rotate(2,d(9));
    for i=38:39
       x(i,:)=rpalm*rthumb*rthumbsecond*(std(i,:)'-std(37,:)')+othumbsecond;
    end
    
    othumbthird=x(39,:)';
    rthumbthird=rotate(2,d(10));
    for i=40:41
       x(i,:)=rpalm*rthumb*rthumbsecond*rthumbthird*(std(i,:)'-std(39,:)')+othumbthird;
    end
    
    
    
    
%%%%%%First Joint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    %%%Index finger
    p=26; %rotation sphere
    o=x(p,:)';
    rindex=rotate(0,d(11))*rotate(2,d(12));
    for i=42:43
        x(i,:)=rpalm*rindex*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Middle finger
    p=27; %rotation sphere
    o=x(p,:)';
    rmiddle=rotate(0,d(15))*rotate(2,d(16));
    for i=47:48
        x(i,:)=rpalm*rmiddle*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Ring finger
    p=28; %rotation sphere
    o=x(p,:)';
    rring=rotate(0,d(19))*rotate(2,d(20));
    for i=52:53
        x(i,:)=rpalm*rring*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Little finger
    p=25; %rotation sphere
    o=x(p,:)';
    rlittle=rotate(0,d(23))*rotate(2,d(24));
    for i=57:57
        x(i,:)=rpalm*rlittle*(std(i,:)'-std(p,:)')+o;
    end
    %%%

    
    
    
%%%%%%%Second Joint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    %%%Index finger
    p=43; %rotation sphere
    o=x(p,:)';
    rindexsecond=rotate(0,d(13));
    for i=44:45
        x(i,:)=rpalm*rindex*rindexsecond*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Middle finger
    p=48; %rotation sphere
    o=x(p,:)';
    rmiddlesecond=rotate(0,d(17));
    for i=49:50
        x(i,:)=rpalm*rmiddle*rmiddlesecond*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Ring finger
    p=53; %rotation sphere
    o=x(p,:)';
    rringsecond=rotate(0,d(21));
    for i=54:55
        x(i,:)=rpalm*rring*rringsecond*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Little finger
    p=57; %rotation sphere
    o=x(p,:)';
    rlittlesecond=rotate(0,d(25));
    for i=58:58
        x(i,:)=rpalm*rlittle*rlittlesecond*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    
    
%%%%%%Third Joint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    %%%Index finger
    p=45; %rotation sphere
    o=x(p,:)';
    rindexthird=rotate(0,d(14));
    for i=46:46
        x(i,:)=rpalm*rindex*rindexsecond*rindexthird*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Middle finger
    p=50; %rotation sphere
    o=x(p,:)';
    rmiddlethird=rotate(0,d(18));
    for i=51:51
        x(i,:)=rpalm*rmiddle*rmiddlesecond*rmiddlethird*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Ring finger
    p=55; %rotation sphere
    o=x(p,:)';
    rringthird=rotate(0,d(22));
    for i=56:56
        x(i,:)=rpalm*rring*rringsecond*rringthird*(std(i,:)'-std(p,:)')+o;
    end
    %%%
    
    %%%Little finger
    p=58; %rotation sphere
    o=x(p,:)';
    rlittlethird=rotate(0,d(26));
    for i=59:60
        x(i,:)=rpalm*rlittle*rlittlesecond*rlittlethird*(std(i,:)'-std(p,:)')+o;
    end
    %%%    
    
    
    for i=1:60
        x(i,1)=x(i,1)-0.085;
        x(i,2)=x(i,2)-0.43;
        x(i,3)=x(i,3)+2.0;
    end
end




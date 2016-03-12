function [x]=model(d)
    fp=fopen('initialHandParameter','r');
    for i=1:48
        for j=1:3
            std(i,j)=fscanf(fp,'%lf',1);
            std(i,j)=std(i,j)/100.0;
        end
        fscanf(fp,'%lf',1);
    end
    fclose(fp);
    % 1-16: palm
    % 17-24: thumb
    % 25-30: index
    % 31-36: middle
    % 37-42: ring
    % 43-48: little
    o0=d(1:3)';
    r0=rotate(0,d(4))*rotate(1,d(5))*rotate(2,d(6));
    for i=1:16
        x(i,:)=r0*(std(i,:)')+o0;
    end

    o16=x(1,:)'; % rotation joint is sphere 1
    r16=rotate(1,d(7))*rotate(2,d(8));
    for i=17:20
       x(i,:)=r0*r16*(std(i,:)'-std(1,:)')+o16;
    end

    o20=x(20,:)';
    r20=rotate(2,d(9));
    for i=21:22
       x(i,:)=r0*r16*r20*(std(i,:)'-std(20,:)')+o20;
    end
    
    o22=x(22,:)';
    r22=rotate(2,d(10));
    for i=23:24
       x(i,:)=r0*r16*r20*r22*(std(i,:)'-std(22,:)')+o22;
    end
    
    for k=1:4
       p=12+k;  % rotation sphere is 13,14,15,16
       o=x(p,:)'; % set p
       r1=rotate(0,d(11+4*k-4))*rotate(2,d(12+4*k-4));
       for i=25+k*6-6:25+k*6-6+1
        x(i,:)=r0*r1*(std(i,:)'-std(p,:)')+o;
       end

       p=25+k*6-6+1;
       o=x(p,:)';
       r2=rotate(0,d(13+4*k-4));
       for i=p+1:p+2
          x(i,:)=r0*r1*r2*(std(i,:)'-std(p,:)')+o; % bug fix
       end

       p=p+2;
       o=x(p,:)';
       r3=rotate(0,d(14+4*k-4));
       for i=p+1:p+2
          x(i,:)=r0*r1*r2*r3*(std(i,:)'-std(p,:)')+o; % bug fix
       end
    end
    for i=1:48
        x(i,1)=x(i,1)-0.2;
        x(i,2)=x(i,2)-0.7;
        x(i,3)=x(i,3)+2.5;
    end
end




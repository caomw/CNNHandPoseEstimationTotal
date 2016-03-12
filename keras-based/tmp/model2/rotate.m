function [R]=rotate(t,d)
if (t==0)
    R=[1 0 0;0 cos(d) -sin(d); 0 sin(d) cos(d)];
elseif (t==1)
    R=[cos(d) 0 -sin(d);0 1 0;sin(d) 0 cos(d)];
elseif (t==2)
    R=[cos(d) -sin(d) 0;sin(d) cos(d) 0;0 0 1];
end

end
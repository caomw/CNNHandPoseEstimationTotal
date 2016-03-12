clear
rng;
d=zeros(26);

low=[ -1, -1, -1,-pi/4, -pi/2, -pi/2,0, 0, 0, 0,-pi/9, -pi/9, 0, 0,-pi/9, -pi/18, 0, 0,-pi/9, -pi/18, 0, 0,-pi/9, -pi/18, 0, 0];
up=[ 1, 1, 1,pi/2, pi/2, pi/2,pi/2, pi/2, pi/2, pi/2,pi/2, pi/18, pi/2, pi/2,pi/2, pi/18, pi/2, pi/2,pi/2, pi/18, pi/2, pi/2,pi/2, pi/9, pi/2, pi/2];
d=[0.038 0.0807 -0.52 0.45 -0.86 0.07 0.18 0.40 1.02 1.55 0.49 0.01 0.067 0.017 0.0603 -0.15 0.1158 0.367 -0.0806 -0.20 -0.0018 0.05 -0.36 -0.04 -0.016 0.546];


d=zeros(26);
x=model(d);


for i=1:16
   plot3(x(i,1),x(i,2),x(i,3),'b*')
   hold on;
   axis equal;
end
for i=17:24
   plot3(x(i,1),x(i,2),x(i,3),'r*')
   hold on;
end
for i=25:30
   plot3(x(i,1),x(i,2),x(i,3),'g*');
   hold on;
end
for i=31:36
   plot3(x(i,1),x(i,2),x(i,3),'y*');
   hold on;
end
for i=37:42
   plot3(x(i,1),x(i,2),x(i,3),'m*');
   hold on;
end
for i=43:48
   plot3(x(i,1),x(i,2),x(i,3),'k*');
   hold on;
end

clear;clc;

W=[1,1,1,1,1];
X=[19,25,31,38,44];
Y=[19,32.3,49,73.3,97.8];
n=2;
a=MyLSF(X,Y,W,n);
b=polyfit(X,Y,1);
c=polyfit(X,Y,2);

x=0:1:50;
y_1=a(1)+a(2)*x.^2;
y_2=polyval(b,x);
y_3=polyval(c,x);

subplot(1,3,1);
plot(X,Y,'*');
hold on;
plot(x,y_1);

subplot(1,3,2);
plot(X,Y,'*');
hold on;
plot(x,y_2);

subplot(1,3,3);
plot(X,Y,'*');
hold on;
plot(x,y_3);






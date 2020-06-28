clear;clc;
A=[4,-1,0;-1,4,-1;0,-1,4];
b=[1;4;-3];
w=1.1;
error_limit=5*10^(-6);
x_gauss=MyGauss(A,b);
x_itera=MySOR(A,b,w,x_gauss,error_limit);

clc;clear;
%init
h=[0.02,0.01];
X=0:0.02:2;
%X=0:0.5:20;
y_True=(exp(-50*X)/2)+X.^2;
y_Euler=zeros(length(X),1);
y_RK=zeros(length(X),1);
y_Euler(1)=1/2;
y_RK(1)=1/2;
for j=1:length(h)
    for i=2:length(X)
        %Euler
        y_Euler(i)=2*h(j)*X(i-1)*(25*X(i-1)+1)+(1-50*h(j))*y_Euler(i-1);
        %Runge-Kutta
        %K1=f(x_n,y_n)
        K1=-50*y_RK(i-1)+50*X(i-1)^2+2*X(i-1);
        %K2=f(x_n+h/2,y_n+h*K1/2)
        K2=-50*(y_RK(i-1)+h(j)*K1/2)+50*(X(i-1)+h(j)/2)^2+2*(X(i-1)+h(j)/2);
        %K3=f(x_n+h/2,y_n+h*K2/2)
        K3=-50*(y_RK(i-1)+h(j)*K2/2)+50*(X(i-1)+h(j)/2)^2+2*(X(i-1)+h(j)/2);
        %K4=f(x_n+h,y_n+h*K3)
        K4=-50*(y_RK(i-1)+h(j)*K3)+50*(X(i-1)+h(j))^2+2*(X(i-1)+h(j));
        %y_(n+1)=y_n+(h/6)*(K1+2K2)
        y_RK(i)=y_RK(i-1)+(h(j)/6)*(K1+2*K2+2*K3+K4);
    end
    subplot(1,2,j);
    plot(X,y_True);
    hold on;
    plot(X,y_Euler);
    hold on;
    plot(X,y_RK);    
end
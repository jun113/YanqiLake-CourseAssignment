function x=MyGauss(A,b)
    n=length(A);
    m=numel(A)/n;
    x=zeros(n,1);
    for i = 1:n
        if (A(i,i)==0)
            disp('wrong input!');
            return;
        end
        for j = i+1:m
            r=A(j,i)/A(i,i);
            A(j,i:n)=A(j,i:n)-r*A(i,i:n);
            b(j)=b(j)-r*b(i);
        end
    end
    x(n)=b(n)/A(n,n);
    for i=n-1:-1:1
        x(i)=(b(i)-sum(A(i,i+1:n)*x(i+1:n)))/A(i,i);
    end
    disp('Gauss elimination:');
    disp(x.');
function x_k1=MySOR(A,b,w,x_gauss,error_limit)
    n=length(A);
    m=numel(A)/n;
    x_k=zeros(n,1);
    x_k1=zeros(n,1);
%    for iter=1:1:inf
    disp('Successive Over Relaxation:');
    disp(['w= ',num2str(w)]);
    for iter=1:1:150
        for i=1:1:n
            sum_1=0;
            sum_2=0;
            for j=1:1:n
                if((j>=1) && (j<=(i-1)))
                    sum_1=sum_1+A(i,j)*x_k1(j);
                end
                if((j>=i) && (j<=n))
                    sum_2=sum_2+A(i,j)*x_k(j);
                end
            end
            x_k1(i)=x_k(i)+w*(b(i)-sum_1-sum_2)/A(i,i);
        end
        error_value=x_k1-x_gauss;
        error_norm=norm(error_value,inf);
        disp(['iterations=',num2str(iter)]);
        disp(x_k1.');
        disp('-------------------');
        if (error_norm <= error_limit)
            disp(['iterations= ',num2str(iter),'   error_norm= ',num2str(error_norm)]);
            disp('iteration done.');
            return
        end
        x_k=x_k1;
    end

    
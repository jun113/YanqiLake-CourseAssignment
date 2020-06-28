function [output] = MyLSF(X,Y,W,n)
    L=length(W);
    G=ones(n,n);
    D=ones(n,1);
    PHI=ones(n,L);

    for i=1:L
        PHI(2,i)=X(i)^2;
    end

    for i=1:n
        for j=1:n
           G(i,j)=MySum(PHI(i,1:L),PHI(j,1:L),W); 
        end
        D(i)=MySum(PHI(i,1:L),Y,W);
    end
    output=G\D;
end


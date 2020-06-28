function x=MySum(a,b,w)
    k=length(a);
    x=0;
    for i=1:k
        x=x+w(i)*a(i)*b(i);
    end

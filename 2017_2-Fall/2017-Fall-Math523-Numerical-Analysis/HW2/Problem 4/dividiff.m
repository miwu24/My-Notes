function [c] = dividiff(x,y)
[n,m]=size(y);
if n==1, n=m;end
n = n-1;
d=zeros(n+1,n+1);
d(:,1)=y;
for j=2:n+1
    for i=j:n+1
        d(i,j)=(d(i-1,j-1)-d(i,j-1))/(x(i-j+1)-x(i));
    end
end
c = diag(d);
end
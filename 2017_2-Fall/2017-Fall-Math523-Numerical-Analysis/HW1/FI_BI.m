function [x] = FI_BI(A,b,L,U,P)
%%%%% This section is doing the forward interation
b_L = P*b;
y = zeros(size(A,1),1);
y(1) = b_L(1)/L(1,1);
for k = 2:size(A,1)
y(k) = (b_L(k)-L(k,1:k-1)*y(1:k-1))/L(k,k);
end

%%%%% This section is doing the backward iteration
m = size(A,1);
x = zeros(m,1);
x(m) = y(m)/U(m,m);
for k = 2:m
    x(m-k+1) = (y(m-k+1)-U(m-k+1,m-k+2:m)*x(m-k+2:m))/U(m-k+1,m-k+1);
end
end
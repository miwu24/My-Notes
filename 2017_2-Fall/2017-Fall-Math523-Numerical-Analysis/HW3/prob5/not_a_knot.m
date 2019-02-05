%%% Cubic spline with not-a-knot boundary conditions.
%%% The nodes are assumed to be equally spaced.
%%% [a, b] = the interval
%%% N = n+1 = the number of nodes
%%% y = the values [f_0, ..., f_n]^T.
function [c] = not_a_knot(a,b,N,y)
n = N-1;
mu = 1/2; lambda = 1/2; h = (b-a)/(N-1);
d = [];
for i = 1:n-1
    d_i = 6/(2*h)*((y(i+2)-y(i+1))/h-(y(i+1)-y(i))/h);
    d = [d;d_i];
end
A = zeros(N,N);
A(1,1) = lambda; A(1,2) = -1; A(1,3) = mu;
A(N,N-2) = lambda; A(N,N-1) = -1; A(N,N) = mu;
for i = 1:n-1
    A(i+1,i) = mu;
    A(i+1,i+1) = 2;
    A(i+1,i+2) = lambda;
end
d_adj = [0;d;0];
M = A\d_adj; %%% [M_0, ..., M_n]^T
c = zeros(n,4);
for i = 0:n-1
    C_tilde_i = y(i+1)-M(i+1)*h^2/6; 
    C_i = (y(i+2)-y(i+1))/h - (h/6)*(M(i+2)-M(i+1));
    c(i+1,1) = M(i+1)*(h^3)/(6*h) + C_tilde_i;
    c(i+1,2) = M(i+1)*(-1)*h^2/(2*h) + C_i;
    c(i+1,3) = M(i+1)*h/(2*h);
    c(i+1,4) = -M(i+1)/(6*h) + M(i+2)/(6*h);
end
end




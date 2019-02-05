function [q] = trapezoid(f,a,b,n)
%TRAPEZOID Summary of this function goes here
%   Detailed explanation goes here
X = linspace(a,b,n+1);
h = (b-a)/n;
f_X = arrayfun(@(x) feval(f,x),X);
q = h/2*dot(f_X,[1,2*ones(1,n-1),1]);

end


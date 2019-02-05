function [t] = ChebEval(n,x)
%CHEBEVAL Summary of this function goes here
%   Detailed explanation goes here
T = [1;x];
if n>=2
    for i = 2:n
       T_i_plus_1 = 2*x*T(i)-T(i-1);
       T = [T;T_i_plus_1];
    end
end
t = T(n+1);
end


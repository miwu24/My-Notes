function [x] = Solve(A,b)
%SOLVE Summary of this function goes here
%   Detailed explanation goes here
[P,L,U] = LU(A);
[x] = FI_BI(A,b,L,U,P);
end


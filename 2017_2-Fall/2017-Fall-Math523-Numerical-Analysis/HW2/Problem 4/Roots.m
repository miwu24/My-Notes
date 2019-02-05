%% This function computes the kth root of the Chebyshev polynomial T_n
function [R] = Roots(n,k)
R = cos((2*k-1)*pi/(2*n));
end
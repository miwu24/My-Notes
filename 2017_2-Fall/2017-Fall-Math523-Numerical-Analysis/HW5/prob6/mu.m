function [ mu_N ] = mu( N )
%MU Summary of this function goes here
%   Detailed explanation goes here
phi_N = phi_matrix(N);
mu_N = [];
for k=0:N
    mu_N = [mu_N;max(abs(phi_N(k+1,:)))];
end

end


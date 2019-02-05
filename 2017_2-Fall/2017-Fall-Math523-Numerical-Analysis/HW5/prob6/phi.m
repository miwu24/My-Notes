function [ phi_N_x ] = phi(N,x)
%PHI Summary of this function goes here
%   Detailed explanation goes here
phi = [0;1];
b = beta(N);
for k = 0:N-1
phi_k_1 = (x-0)*phi(k+2)-b(k+1)*phi(k+1);
phi = [phi;phi_k_1];
end
phi_N_x = phi(2:end);
end


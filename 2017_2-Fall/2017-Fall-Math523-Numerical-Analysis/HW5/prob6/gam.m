function [ gam_N ] = gam( N )
%GAMMA Summary of this function goes here
%   Detailed explanation goes here
gam_N = [];
phi_N = phi_matrix(N);
for k=0:N
    gam_N = [gam_N; norm(phi_N(k+1,:))^2];
end
gam_N = (2/(N+1))*gam_N;
end


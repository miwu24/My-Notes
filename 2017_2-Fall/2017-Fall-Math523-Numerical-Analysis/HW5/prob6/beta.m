function [b] = beta( N )
%BETA Summary of this function goes here
%   Detailed explanation goes here
K = [1:N];
b = [2,(1+1/N)^2*(1-(K/(N+1)).^2)./(4-1./K.^2)]';
end


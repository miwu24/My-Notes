function [val] = humps(x)
%HUMPS Summary of this function goes here
%   Detailed explanation goes here
val = 1/((x-0.3)^2+0.001)+1/((x-0.9)^2+0.04);
end


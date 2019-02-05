function [y] = berstein(n,j,t)
%BERSTEIN Summary of this function goes here
%   computes the value of the jth berstein polynomial of degree n at t
y = nchoosek(n,j)*t^j*(1-t)^(n-j);
end


%% This function computes the polynomial evaluation using Horner's rule
function [v] = Horner(c, x, z)
    c_flip = flip(c);
    x_flip = flip(x);
    v = c_flip(1);
    for i=1:length(c)-1
        v = v*(z-x_flip(i+1)) + c_flip(i+1);
    end
function [ f_x ] = fun_prob5( x )
%FUN_PROB5 Summary of this function goes here
%   Detailed explanation goes here
if (0<=x)&&(x<=pi)
    f_x = (x/pi)^2-x/pi;
elseif (pi<=x)&&(x<=2*pi)
    f_x = (x-pi)/pi-((x-pi)/pi)^2;
end

end


%%% This function evaluate the Chebyshev polynomial of degree n at x
function [t] = Chebyshev(n,x)
if max(size(x))==1
    t = ChebEval(n,x);
else
    t = arrayfun(@(xx) ChebEval(n,xx),x);
end
end



    
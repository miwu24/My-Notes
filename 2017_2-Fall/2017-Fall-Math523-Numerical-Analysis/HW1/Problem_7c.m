function [OneNormErr,TwoNormErr] = Problem_7c(m)
[a,y] = Problem_7b(m);
PolyVal = polyval(flip(a),y);
SinVal = sin(pi*y);
OneNormErr = sum(abs(PolyVal-SinVal));
TwoNormErr = sqrt(sum((PolyVal-SinVal).^2));
end


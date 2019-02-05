%% Test whether Horner's rule works as expected with the example in Problem_1a
Problem_1a
V = [];
for i=1:length(x)
    [v] = Horner(c,x,x(i));
    V = [V,v];
end

if V==y
    disp('Horner works as expected')
else
    disp('The code of Horner has some problem')
end
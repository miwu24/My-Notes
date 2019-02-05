function [x,y] = PWLAdapt(fname,xL,xR,delta,hzero)
x = [xL;xR];
y = arrayfun(@(t) feval(fname,t),x);
mid = (xL+xR)/2;
DelValue = abs(feval(fname,mid)-mean([feval(fname,xL);feval(fname,xR)]));
DelX = xR-xL;
Test = [];
if ~((DelValue<=delta)|(DelX<=hzero))
    Test = [Test;1];
end
Test;
while length(Test)~=0
    Test = [];
    x_copy = [x(1)];
    for i=1:length(x)-1
        mid = (x(i)+x(i+1))/2;
        DelValue = abs(feval(fname,mid)-mean([feval(fname,x(i));feval(fname,x(i+1))]));
        DelX = x(i+1)-x(i);
        if ~((DelValue<=delta)|(DelX<=hzero))
            Test = [Test;1];
            x_copy = [x_copy; mid; x(i+1)];
        else
            x_copy = [x_copy; x(i+1)];
        end
    end
    x = x_copy;            
end
x;
y = arrayfun(@(t) feval(fname,t),x);
end
    



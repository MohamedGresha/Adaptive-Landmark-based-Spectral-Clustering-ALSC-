function [Xscaled,varmin,varmax,varrange]=atscale(X)
%scale an input data to be from 0->1 for each variable.
% Input:
%       X: and nxd data matrix.
% Output:
%       Xscaled: and nxd scaled data matrix.
%       varmin: a 1xd vector of column minimums.
%       varrange: a 1xd vector of column ranges.

% AhmedToolBox Aug 2003

[n,d]=size(X);
varmin=zeros(1,d);
varmax=zeros(1,d);
varrange=zeros(1,d);
Xscaled=zeros(n,d);

% scaling
for i=1:d
    varmin(i)=min(X(:,i));
    varmax(i)=max(X(:,i));
    varrange(i)=varmax(i)-varmin(i);
    
    if(varmax(i)>0)
        Xscaled(:,i)=(X(:,i)-varmin(i))./varrange(i);
    else
        Xscaled(:,i)=0;
    end
end
return;

function [bestk,clusterid,NMI]=demo3(k)
% this is demo3.m a function tests differnet algorithms in estimating the number
% of clusters of the Iris data set.
%
%NMI is the normalized mutual information between the cluster id vector and the class id vector.

%Ahmed Rafat
%MAR. 13
%
% Updated by 
% Mohamed Gresha
% 2021

load USPS;

y = X';
X=y';
[X,varmin,varmax,varrange]=atscale(X);
y=X';

%% Select the algorithm
%[bestk,bestmu,bestcov,bestpp,clusterid] = mixtures4(y);        %Figueiredo 2002********
%[bestk,bestmu,bestcov,bestpp,clusterid]=atembic(y');        %BIC
%[bestk,bestmu,bestcov,bestpp,clusterid]=atemmi(y');         %MI***********
%[bestk,bestmu,bestcov,bestpp,clusterid]=atemmi_m(y');   %MI+CEM_Modified******
%[bestk,bestmu,bestcov,bestpp,MIPL,clusterid]=atemmipl(y');   %MI+CEM_PL******
%[bestk,bestw,ACL,clusterid,ACL_vec,k_vec]=atacl(y');      % Competitive neural network


%[bestk,bestw,ACL,clusterid,NN]=atacl_n(y',k);
%[clusterid] = kmeans(y', k);
clusterid = LSC(y',k);
bestk=k;
%[bestk,bestw,ACL,clusterid,NN]=atacl_parfor(y');

% compute the normalized mutual information
if bestk == 1
    NMI = 0;
else
    
    classid=[1*ones(1553,1);
        2*ones(1269,1);
        3*ones(929,1);
        4*ones(824,1);
        5*ones(852,1);
        6*ones(716,1);
        7*ones(834,1);
        8*ones(792,1);
        9*ones(708,1);
        10*ones(821,1)
        ];
    pclass = [1553/9298 1269/9298 929/9298 824/9298 852/9298 716/9298 834/9298 792/9298 708/9298 821/9298];   %the probability of each class
    pcluster=[];                %bestpp;        %the probability of each cluster
    pclass_cluster=[];      %the probability that a member of cluster j belongs to class i
    n = length(classid);
    for i=1:10
        if i==1
            c1=1;
            c2=1553;
        elseif i==2
            c1=1554;
            c2=2822;
        elseif i==3
            c1=2823;
            c2=3751;
        elseif i==4
            c1=3752;
            c2=4575;
        elseif i==5
            c1=4576;
            c2=5427;
        elseif i==6
            c1=5428;
            c2=6143;
        elseif i==7
            c1=6144;
            c2=6977;
        elseif i==8
            c1=6978;
            c2=7769;
        elseif i==9
            c1=7770;
            c2=8477;
        elseif i==10
            c1=8478;
            c2=9298;
        end
        for j=1:bestk
            nj=length(find(clusterid == j));
            nij=length(find(clusterid(c1:c2) == j));
            pclass_cluster(i,j)=nij/n;
            pcluster(j)= nj/n;
        end
    end
    [g,h]=size(pcluster);
    for w=1:h
        if pcluster(w)== 0
            pcluster(w)= 0.000001;
        end
    end
    Hclass = -sum(pclass.*log(pclass)/log(2));
    Hcluster = -sum(pcluster.*log(pcluster)/log(2));
    
    MI=0;       % the Mutual information
    for i=1:10
        for j=1:bestk
            if pclass_cluster(i,j) ~= 0
                MI = MI + pclass_cluster(i,j) * log(pclass_cluster(i,j)/(pclass(i) * pcluster(j)))/log(2);
            end
        end
    end
    NMI = MI / sqrt(Hclass * Hcluster);
end
return;

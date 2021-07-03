function [bestk,bestw,ACL,clusterid,NMI,NN]=demo16(k)
% this is demo3.m a function tests differnet algorithms in estimating the number
% of clusters .
%
%NMI is the normalized mutual information between the cluster id vector and the class id vector.

%Ahmed Rafat
%MAR. 13
%
% Updated by 
% Mohamed Gresha
% 2021
%
load DATA;

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
% [clusterid] = kmeans(y', k);
% bestk=k;
[bestk,bestw,ACL,clusterid,NN]=atacl_parfor(y');

% compute the normalized mutual information
if bestk == 1
    NMI = 0;
else
    classid=[1*ones(5923,1);
        2*ones(6742,1);
        3*ones(5958,1);
        4*ones(6131,1);
        5*ones(5842,1);
        6*ones(5421,1);
        7*ones(5918,1);
        8*ones(6265,1);
        9*ones(5851,1);
        10*ones(5949,1)
        ];
    pclass = [5923/60000 6742/60000 5958/60000 6131/60000 5842/60000 5421/60000 5918/60000 6265/60000 5851/60000 5949/60000];   %the probability of each class
    pcluster=[];                %bestpp;        %the probability of each cluster
    pclass_cluster=[];      %the probability that a member of cluster j belongs to class i
    n = length(classid);
    for i=1:10
        if i==1
            c1=1;
            c2=5923;
        elseif i==2
            c1=5924;
            c2=12665;
        elseif i==3
            c1=12666;
            c2=18623;
        elseif i==4
            c1=18624;
            c2=24754;
        elseif i==5
            c1=24755;
            c2=30596;
        elseif i==6
            c1=30597;
            c2=36017;
        elseif i==7
            c1=36018;
            c2=41935;
        elseif i==8
            c1=41936;
            c2=48200;
        elseif i==9
            c1=48201;
            c2=54051;
        elseif i==10
            c1=54052;
            c2=60000;
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

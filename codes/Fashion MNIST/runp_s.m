% run an algorithm to estimate the number of clusters in a data set
% for 100 times and save the results

% Ahmed Rafat
% Mar. 13
%
% Updated by 
% Mohamed Gresha
% 2021
%
% 
kopt=[];
NMI_vec=[];
bestk_vec=[];
ACLopt=[];

base_path = 'D:\1-Paper\Paper 3\11\MNIST\0';
for q=1:10                %run the Algorithm 10 time to
    
    tic
    
    [bestk,bestw,ACL,clusterid,NMI,NN]=demo16(16);     %sequention processing    
    
    time=toc;
    
%     kopt=[kopt bestk];
%     NMI_vec=[NMI_vec;NMI];
%     bestk_vec=[bestk_vec;bestk];
%     ACLopt=[ACLopt;ACL];
    
    SS(q).NN=NN;
    SS(q).kopt=bestk;
    SS(q).NMI=NMI;
    SS(q).bestk=bestk;
    SS(q).ACL=ACL;
    SS(q).time=time;
    
    %     NMI_vec=[NMI_vec;NMI];
    %     bestk_vec=[bestk_vec;bestk];
    %     ACLopt=[ACLopt;ACL];
    %subfolder = ('New folder2');
    file_name=sprintf('0_acl_SS%d.mat',q);
    save(fullfile(base_path,file_name), 'bestk', 'bestw', 'ACL', 'clusterid', 'NMI','time','NN','SS');
end

save(fullfile(base_path,'0_acl_SS'), 'NN', 'SS');
disp('===========================================');
% disp('kopt=');
% disp(length(find(kopt==3)));
[val,id] = min([SS.ACL]);
%[val,id]=min(ACLopt);
disp('======> Min ACL <======');
disp('ACL=');
disp(val);
disp('kopt=');
disp(SS(id).kopt);
disp('NMIopt=');
disp(SS(id).NMI);
disp('time=');
disp(SS(id).time);
[val,id]=max([SS.NMI]);
disp('======> Max NMI<======');
disp('MAXNMI=');
disp(val);
disp('MaxACL=');
disp(SS(id).ACL);
disp('kopt=');
disp(SS(id).kopt);
disp('time=');
disp(SS(id).time);

return;

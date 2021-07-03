%
% 
% Mohamed Gresha
% 2021

base_path = 'D:\1-Paper\Paper4\11\USPS\0';
load ('USPS.mat');

[X,varmin,varrange]=atscale(X);


%================================================================================================
% st1=1;st2=80001;st3=160001;st4=240001;st5=230001;
st=[1,1554, 2823, 3752, 4576, 5428, 6144, 6978, 7770, 8478];
en=[1553, 2822, 3751, 4575, 5427, 6143, 6977, 7769, 8477, 9298];
% en1=1500;
for i=1 :5000
    
    if st(1)+87>en(1) || st(2)+87>en(2) || st(3)+87>en(3) || st(4)+87>en(4) || st(5)+87>en(5) || st(6)+87>en(6) || st(7)+87>en(7) || st(8)+87>en(8) || st(9)+87>en(9) || st(10)+87>en(10)
        break;
    end
    DD(i).data=X([st(1):st(1)+86,st(2):st(2)+86,st(3):st(3)+86,st(4):st(4)+86,st(5):st(5)+86,st(6):st(6)+86,st(7):st(7)+86,st(8):st(8)+86,st(9):st(9)+86,st(10):st(10)+86],:)';
    
    st=st+87;
    %      st(1)=st(1)+87;
    %     st(2)=st(2)+87;
    %     st(3)=st(3)+87;
end
i=i-1;
parfor q=1:i
    tic
%     a=randi(i,1);
    [bestk,bestw,ACL,clusterid1,NN]=atacl(DD(q).data');      % Competitive neural network
    D_ir(q).NN=NN;
    D_ir(q).bestk=bestk;
    D_ir(q).ACL=ACL;
    D_ir(q).clusterid=clusterid1;
%     D_ir(q).Data_Num=a;
    time1=toc;
    D_ir(q).Time=time1;
    
end
file_name=sprintf('2_acl_D_%d.mat',i);
save(fullfile(base_path,file_name), 'D_ir');
%================================================================================================

[val_ir,idx] = min([D_ir.bestk]);
% if val_ir==10
for j=1:10
    tic
    [bestk,clusterid,NMI]=demo3(val_ir);     %sequention processing
    DD_ir(j).NMI=NMI;
    DD_ir(j).Bestk=bestk;
    DD_ir(j).Clusterid=clusterid;
    ri=rand_index(clusterid,Y,'adjusted');
    ri=ri*100;
    DD_ir(j).ARI=ri;
    time2=toc;
    DD_ir(j).Time=time2;
    file_name=sprintf('3_acl_LSC%d.mat',j);
    save(fullfile(base_path,file_name), 'bestk', 'clusterid','DD_ir');
end
% end

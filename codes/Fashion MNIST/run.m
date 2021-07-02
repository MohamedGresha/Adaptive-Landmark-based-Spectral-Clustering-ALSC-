base_path = 'D:\1-Paper\Paper4\11\Fashion MNIST\0';
load ('DATA.mat');

[X,varmin,varrange]=atscale(X);


%================================================================================================
% st1=1;st2=80001;st3=160001;st4=240001;st5=230001;
st=[1,5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051];
en=[5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000];
% en1=1500;
for i=1 :5000
    
    if st(1)+385>en(1) || st(2)+385>en(2) || st(3)+385>en(3) || st(4)+385>en(4) || st(5)+385>en(5) || st(6)+385>en(6) || st(7)+385>en(7) || st(8)+385>en(8) || st(9)+385>en(9) || st(10)+385>en(10)
        break;
    end
    DD(i).data=X([st(1):st(1)+384,st(2):st(2)+384,st(3):st(3)+384,st(4):st(4)+384,st(5):st(5)+384,st(6):st(6)+384,st(7):st(7)+384,st(8):st(8)+384,st(9):st(9)+384,st(10):st(10)+384],:)';
    
    st=st+385;
    %      st(1)=st(1)+385;
    %     st(2)=st(2)+385;
    %     st(3)=st(3)+385;
end
i=i-1;
parfor q = 1:i
    tic
    a=randi(i,1);
    [bestk,bestw,ACL,clusterid1,NN]=atacl(DD(q).data');      % Competitive neural network
    D_ir(q).NN=NN;
    D_ir(q).bestk=bestk;
    D_ir(q).ACL=ACL;
    D_ir(q).clusterid=clusterid1;
    %     D_ir(q).Data_Num=a;
    time1=toc;
    D_ir(q).Time=time1;
    
end
file_name=sprintf('3_acl_D_%d.mat',i);
save(fullfile(base_path,file_name),'D_ir');
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
    file_name=sprintf('4_acl_LSC%d.mat',j);
    save(fullfile(base_path,file_name), 'bestk', 'clusterid','DD_ir');
end
% end
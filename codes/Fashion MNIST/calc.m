%acl=[];
nmi=[];
time=[];
kopt=[];

%seq. calc.
for i=1:10
%acl=[acl PP_SS(i).ACL];
nmi=[nmi PP_SS(i).NMI];
time=[time PP_SS(i).time];
kopt=[kopt PP_SS(i).kopt];
end 
%mean_acl_s=mean(acl);
mean_nmi_s=mean(nmi);
mean_time_s=mean(time);
mean_kopt_s=mean(kopt);
%STD_acl_s=std(acl);
STD_nmi_s=std(nmi);
STD_time_s=std(time);
STD_kopt_s=std(kopt);

%acl=[];
nmi=[];
time=[];
kopt=[];
% Parallel calc.
for i=1:10
%acl=[acl PP_PP(i).ACL];
nmi=[nmi PP_PP(i).NMI];
time=[time PP_PP(i).time];
kopt=[kopt PP_PP(i).kopt];
end 
%mean_acl_p=mean(acl);
mean_nmi_p=mean(nmi);
mean_time_p=mean(time);
mean_kopt_p=mean(kopt);
%STD_acl_p=std(acl);
STD_nmi_p=std(nmi);
STD_time_p=std(time);
STD_kopt_p=std(kopt);
spd=mean_time_s/mean_time_p;
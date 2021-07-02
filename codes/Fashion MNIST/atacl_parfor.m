function [bestk,bestw,bestACL,bestclusterid,NN]=atacl_parfor(data)
%ACL the adaptive competitive neural network that adapts the number of neurons in the output layer based on the input data.
% input:
%       data : is a (n,d) matrix
% output:
%       bestk : is the number of neurons in the output layer,
%       bestw : is a (d,k) matrix in which each neuron weights vector is a column,
%       clusterid : the class ID for each pattern in the data,
%       ACL : adaptive competitive learning criterion value for the CNN

% Ahmed Tool Box
% MAR. 2013

[n,d]=size(data);
% clusterid=zeros(1,n);
y=data';

% ACL_vec=[];
% k_vec=[];
% pp_vec=[];
% MSE_vec=[];
% w_vec=[];
% clusterid_vec=[];
% Competitive Learning
% Neurons in a competitive layer learn to represent different regions of the
% input space where input vectors occur.
%  Syntax
%
%     net = newc(P,S,KLR,CLR)
%
%    Description
%
%      Competitive layers are used to solve classification
%      problems.
%
%      NET = NEWC(P,S,KLR,CLR) takes these inputs,
%        P  - RxQ matrix of Q input vectors.
%        S  - Number of neurons.
%        KLR - Kohonen learning rate, default = 0.01.
%        CLR - Conscience learning rate, default = 0.001.
%      Returns a new competitive layer.

% Create P.
% P = y;      % d x n

% % Plot P.
% plot(P(1,:),P(2,:),'+r');
% title('Input Vectors');
% xlabel('p(1)');
% ylabel('p(2)');

%%
% Here NEWC takes three input arguments, an Rx2 matrix of min and max values for
% R input elements, the number of neurons, and the learning rate.
%
% We can plot the weight vectors to see their initial attempt at classification.
% The weight vectors (o's) will be trained so that they occur centered in
% clusters of input vectors (+'s).
k=16;
%net = newc(P,k);           % newc([0 1;0 1],10);
%w = net.IW{1};
% plot(P(1,:),P(2,:),'+r');
% hold on;
% circles = plot(w(:,1),w(:,2),'ob');


%%
% Set the number of epochs to train before stopping and train this competitive
% layer (may take several seconds).
%
% Plot the updated layer weights on the same graph.

%net.trainParam.epochs = 10;
%net = train(net,P);
%w = net.IW{1};
% delete(circles);
% plot(w(:,1),w(:,2),'ob');


%%
% Now we use the competitive layer as a classifier, where each neuron
% corresponds to a different category.
%
% The output, a, indicates which neuron is responding to each input vector, and thereby which class
% the input belongs. Note that SIM returns outputs in sparse matrix form for
% competitive layers.
%bestw = [];
% a=zeros(k,n);            %a=[];
% ac=zeros(1,n);          %ac=[];
% wts=[];

% a = sim(net,P);
% ac = vec2ind(a);        %1 x n
% clusterid = ac(1:n);
% ac = clusterid;
% wts = net.IW{1,1};  % k x d
% [k,d] = size(wts);

% compute pies + MSE + ACL
% pies=zeros(1,k);
% for i=1:k
%    pies(i) = length(find(ac==i))/n ;
% end
%
% MSE=0;
% for i=1:n
%     for j=1:k
%         if(ac(i)==j)
%             for l=1:d
%                 MSE = MSE +(P(l,i) - wts(j,l))^2;
%             end
%         end
%     end
% end
% MSE = MSE/n;
%
% logpies = 0;
% for i = 1:k
%     if(pies(i)~=0)
%         logpies = logpies + log10(1/pies(i));
%     else
%         logpies = logpies + log10(1/0.001);
%     end
% end
%
% ACL = MSE + 0.05*logpies;

% the current model is the best one
%bestpp = pies;
%bestMSE =  MSE;
%bestw = wts';       % d x k
%bestACL = ACL;
%bestk = k;
%bestclusterid = clusterid;
%ACL_vec = [ACL_vec ACL];
%k_vec=[k_vec k];
%
%     ACL_vec = [ACL_vec ACL];
%     k_vec=[k_vec k];
%     pp_vec=[pp_vec pies];
%     MSE_vec=[MSE_vec MSE];
%     w_vec=[w_vec wts'];
%     clusterid_vec=[clusterid_vec clusterid];

parfor ki = 1:k
    P = y;      % d x n

    %while(k > 1)
    %   k=k-1;
    tic
    net = newc(P,ki);            %newc([0 1;0 1],k);
    net.trainParam.epochs = 10;
    net = train(net,P);
    
    a=zeros(ki,n);            %a=[];
    ac=zeros(1,n);          %ac=[];
    wts=[];
    
    a = sim(net,P);
    ac = vec2ind(a);        %1 x n
    clusterid = ac(1:n);
    ac = clusterid;
    wts = net.IW{1,1}  % k x d
    
    % compute pies + MSE + ACL
    pies=zeros(1,ki);
    for i=1:ki
        pies(i) = length(find(ac==i))/n ;
    end
    
    MSE=0;
    for i=1:n
        for j=1:ki
            if(ac(i)==j)
                for l=1:d
                    MSE = MSE +(P(l,i) - wts(j,l))^2;
                end
            end
        end
    end
    MSE = MSE/n;
    
    logpies = 0;
    for i = 1:ki
        if(pies(i)~=0)
            logpies = logpies + log10(1/pies(i));
        else
            logpies = logpies + log10(1/0.001);
        end
    end
    
    ACL = MSE + 0.05*logpies;
    disp('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*');
    ki
    MSE
    logpies
    c=logpies*0.05;
    c
    disp('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*');
    
    NN(ki).ACL       = ACL;
    NN(ki).K         = ki;
    NN(ki).PP        = pies;
    NN(ki).MSE       = MSE;
    NN(ki).logpies   = logpies;
    NN(ki).logpies1  = logpies*0.05;
    NN(ki).wts       = wts';
    NN(ki).clusterid = clusterid;
    time1=toc
    NN(ki).time      = time1;
    %      ACL_vec      =[ACL_vec ACL];
    %      k_vec        =[k_vec ki];
    %      pp_vec       =[pp_vec pies];
    %      MSE_vec      =[MSE_vec MSE];
    %      w_vec        =[w_vec wts'];
    %      clusterid_vec=[clusterid_vec clusterid];
    
    %     if ACL<bestACL
    %         % the current model is the best one
    %         bestpp = pies;
    %         bestMSE =  MSE;
    %         bestw = wts';       % d x k
    %         bestACL = ACL;
    %         bestk = k;
    %         bestclusterid = clusterid;
    %     end
end %end of while(k>1)
% NN.ACL
% NN.K
% NN.PP
% NN.MSE
[val,ix] = min([NN.ACL]);
bestpp = NN(ix).PP;
bestMSE =  NN(ix).MSE;
bestw = NN(ix).wts;       % d x k
bestACL = val;
bestk = NN(ix).K ;
bestclusterid = NN(ix).clusterid;
% disp(NN(ix).wts);
%          [a b]=min(ACL_vec);
%          x=size(w_vec);
%          z=size(ACL_vec);
%          c=size(w_vec)/size(ACL_vec);
%          bestpp = pp_vec(b);
%          bestMSE =  MSE_vec(b);
%          bestw = w_vec(:,((c*(b-1))+1):(b+c));       % d x k
%          bestACL = ACL_vec(b);
%          bestk = k_vec(b) ;
%          bestclusterid = clusterid_vec(b);
%plot the best model
% figure
%     plot(y(axis1,:),y(axis2,:),'.','Color',[0.5 0.5 0.5]);
%     hold on
%     axis equal
%     set(gca,'FontName','Times','FontSize',22);
%     placex = get(gca,'Xlim'); placey = get(gca,'Ylim');
%     text(placex(1)+1,placey(2)-1,sprintf('k=%d',length(bestpp)),...
%          'FontName','Times','FontSize',22);
%     drawnow
%     hold off
%thefig=gcf;
return;

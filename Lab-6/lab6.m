%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course: Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student: 陈逸韬
% ID: 2020141460308
%
% Lab 6 - Sequence auto-complete
%
% Task:
% Design a multi-target outputs neural network to learn to complete sequence.
% The first two items of a sequence uniquely determine the remaining four.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear workspace and close plot windows
clear;
close all;
% prepare the data set
train_seq = {
    'AA1212' 'AC1231' 'AD1221' 'AE1213'
    'BA2312' 'BB2323' 'BC2331' 'BE2313'
    'CB3123' 'CC3131' 'CD3121' 'CE3113'
    'DA2112' 'DB2123' 'DC2131' 'DD2121'
    'EA1312' 'EB1323' 'ED1321' 'EE1313'
    };
test_seq = {
    'AB1223' 'BD2321' 'CA3112' 'DE2113' 'EC1331'
    };
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% encode datasets
%%
%%
% prepare training data
% prepare testing data
% choose parameters
max_iter=100;
lr=0.1;
f = @(s) 1 ./ (1 + exp(-s)); 
df = @(s) f(s) .* (1 - f(s)); 
% define the network architecture
N = [5,5,3,3,3,3];
% initialize weights
%%
W = cell(1,4);
for i=1:4
    W{i}=0.1*randn(3,13);
end
A=cell(1,4);
Z=cell(1,4);
X=cell(1,5);
Y=cell(1,4);
delta=cell(1,4);
dw=cell(1,4);
%%
% train
J = [];
Acc = [];
%%
% loop until converge
for iter = 1:max_iter
    % for each mini-batch
    [m,n]=size(train_seq);
    for i = 1:m
        for j=1:n
            x=train_seq{i,j};
            [encode_x,encode_idx]=encode(x);
            % forward computation
            AB=encode_x{1};
            X{1}=[zeros(3,1) ; AB ];
            for ii = 1:4
               Y{ii}=encode_x{ii+1}; 
            end
            for ii = 1:4
                [A{ii} ,Z{ii}]=fc(W{ii},X{ii});
                X{ii+1}=[A{ii};AB];
            end
            % backward computation (need some attention here)
            delta{4}=(A{4}-Y{4}).*df(Z{4});
            dw{4}=delta{4} * X{4}';
            for ii =3:-1:1
               delta{ii}=[df(Z{ii}) ; ones(10,1)] .* W{ii+1}' * delta{ii+1} + [ df(Z{ii}).*(A{ii}-Y{ii}) ; zeros(10,1)];
               delta{ii}=delta{ii}(1:3);
               dw{ii}=delta{ii} * X{ii}';
            end
            
            for ii =1:4
                W{ii}=W{ii}-lr*dw{ii};
            end
            % update weight
            
            % cost function on train batch (sums from all layers)
            loss=0;
            for ii = 1:4
                loss=loss+(1/2)*sum((A{ii}-Y{ii}).^2);
            end
            J = [J loss];
            plot(J,'-b');drawnow
            % accuary on train batch
%             batch_Acc = [];
%             Acc = [Acc batch_Acc];
        end
    end
    % optionally you can display J and Acc on-the-fly
end
plot(1:2000,J,'-b');
save model
save model.mat W n Acc J
%%
% test
% test on training set
total_num=20;num=0;
[m,n]=size(train_seq);
for i = 1:m
    for j=1:n
        x=train_seq{i,j};
        [encode_x,encode_index]=encode(x);
        label=encode_index(3:6);
        AB=encode_x{1};
        X{1}=[zeros(3,1) ; AB ];
        for ii = 1:4
            [A{ii} ,Z{ii}]=fc(W{ii},X{ii});
            X{ii+1}=[A{ii};AB];
        end
        Y_pre=int32(zeros(1,4));
        for ii = 1:4
           [data,index]=max(A{ii});
           Y{ii}=index; 
           Y_pre(ii)=index;
        end
        
         if isequal(label,Y_pre)
             num=num+1;
         end
    end
end
%%
train_acc=num/total_num;
% test on testing set
test_acc =0;
test_num=0
for i = 1:5
    x=test_seq{i};
    [encode_x,encode_index]=encode(x);
    label=encode_index(3:6);
    AB=encode_x{1};
    X{1}=[zeros(3,1) ; AB ];
    for ii = 1:4
        [A{ii} ,Z{ii}]=fc(W{ii},X{ii});
        X{ii+1}=[A{ii};AB];
    end
    Y_pre=int32(zeros(1,4));
    for ii = 1:4
       [data,index]=max(A{ii});
       Y{ii}=index; 
       Y_pre(ii)=index;
    end

     if isequal(label,Y_pre)
         test_num=test_num+1;
     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_acc=test_num/5;
% display results
fprintf('Accuracy on training dataset is %f%%\n', train_acc*100);
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);
%%
B=[];
for i = 1:4
    B=[ B A{i}];
end
%%
disp(label);
disp(B);


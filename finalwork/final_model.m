clear; clc;

load train_final_32x32.mat % 导入训练数据集
load test_final_32x32.mat % 导入测试数据集

disp("初始化参数列表");

% 存储 mini batch的损失
J = []; 
% 存储 mini batch的正确率
Acc = []; 
% 定义最大训练次数
max_epochs = 200;
% mini batch 的大小
mini_batch = 200; 
% 定义全连接层数：7
L = 7;
% 定义全连接之间的维度
layer_size = [1024
        512
        256
        128
        64
        32
        10];

% 初始化学习率
learning_rate = 0.0005;
% 初始化权重矩阵
for l = 1:L - 1
    w{l} = 0.1 * randn(layer_size(l + 1, 1), sum(layer_size(l, :)));
end

% 画图
figure 

for iter = 1:max_epochs % 进行每次大的循环
    fprintf("这次是第%d次训练",iter);

    idxs = randperm(train_data_size);

    for k = 1:ceil(train_data_size / mini_batch)
        start_idx = (k - 1) * mini_batch + 1;
        end_idx = min(k * mini_batch, train_data_size);

        if mod(k, 20) == 0
            percent = end_idx / train_data_size * 100;
            fprintf('No.%d trains has complete %8.4f %% \n', iter, percent);
        end

        a{1} = X_train(:, idxs(start_idx:end_idx));
        y = train_labels(:, idxs(start_idx:end_idx));

        [a{2}, z{2}] = fc_relu(w{1}, a{1});
        [a{3}, z{3}] = fc_relu(w{2}, a{2});
        [a{4}, z{4}] = fc_relu(w{3}, a{3});
        [a{5}, z{5}] = fc_relu(w{4}, a{4});
        [a{6}, z{6}] = fc_relu(w{5}, a{5});
        [a{7}, z{7}] = fc_sigmoid(w{6}, a{6});

        delta{L} = (a{L} - y) .* a{L} .* (1 - a{L});
        delta{6} = bc_relu(w{6}, z{6}, delta{7});
        delta{5} = bc_relu(w{5}, z{5}, delta{6});
        delta{4} = bc_relu(w{4}, z{4}, delta{5});
        delta{3} = bc_relu(w{3}, z{3}, delta{4});
        delta{2} = bc_sigmoid(w{2}, z{2}, delta{3});

        for l = 1:L - 1
            grad_w = delta{l + 1} * a{l}';
            w{l} = w{l} - learning_rate * grad_w;
        end

        J = [J 1 / length(y) * sum(cost_mse(a{L}, y))];
        acc = accuracy(a{L}, y);
        Acc = [Acc acc];
        plot(J);
        pause(0.00001);
    end

    fprintf('No.%d acc is %8.4f %% \n', iter, acc * 100);
end

figure
plot(Acc);
save model.mat w layer_size;

a{1} = X_train;
[a{2}, z{2}] = fc_relu(w{1}, a{1});
[a{3}, z{3}] = fc_relu(w{2}, a{2});
[a{4}, z{4}] = fc_relu(w{3}, a{3});
[a{5}, z{5}] = fc_relu(w{4}, a{4});
[a{6}, z{6}] = fc_relu(w{5}, a{5});
[a{7}, z{7}] = fc_sigmoid(w{6}, a{6});
train_acc = accuracy(a{L}, train_labels);
fprintf('训练集的准确率: %f%%\n', train_acc * 100);

a{1} = X_test;
[a{2}, z{2}] = fc_relu(w{1}, a{1});
[a{3}, z{3}] = fc_relu(w{2}, a{2});
[a{4}, z{4}] = fc_relu(w{3}, a{3});
[a{5}, z{5}] = fc_relu(w{4}, a{4});
[a{6}, z{6}] = fc_relu(w{5}, a{5});
[a{7}, z{7}] = fc_sigmoid(w{6}, a{6});
test_acc = accuracy(a{length(w) + 1}, test_labels);
fprintf('测试集的准确率: %f%%\n', test_acc * 100);


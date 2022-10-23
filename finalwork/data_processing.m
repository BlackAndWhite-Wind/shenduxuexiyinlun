clear; clc;
% 尝试导入train_final_32x32数据集，否则对train_32x32数据集进行预处理
try
    load train_final_32x32.mat
catch
    load train_32x32.mat % 导入数据集

    train_data_size = length(X); % 训练集的长度
    disp("训练集长度大小："); disp(train_data_size);

    train_gray = ones(32, 32, train_data_size); % 初始化转化之后的灰度图像

    for i = 1:train_data_size
        train_gray(:, :, i) = rgb2gray(X(:, :, :, i));
        % 调用rgb2gray函数，把彩色图像变为灰色图像，降低通道数，3->1
    end

    train_gray = train_gray / 255;
    X_train = reshape(train_gray, 32 * 32, train_data_size);
    % X_train为最后训练集
    train_label_size = length(y);
    train_labels = zeros(10, train_label_size);
    % 0-9的数字编号

    for i = 1:train_label_size
        temp = y(i, 1);
        train_labels(temp, i) = 1;
    end
    
    % 保存最终的数据集格式 train_final_32x32.mat
    %          训练数据集 X_train
    %          数据的长度 train_data_size
    %          数据的标签 train_labels
    save train_final_32x32.mat X_train train_data_size train_labels
end

% 尝试导入test_final_32x32数据集，否则对test_32x32数据集进行预处理
try
    load test_final_32x32.mat
catch
    load test_32x32.mat % 导入数据集

    test_data_size = length(X);
    disp("测试集长度大小："); disp(test_data_size);
    test_gray = ones(32, 32, test_data_size); % 初始化转化之后的灰度图

    for i = 1:length(X)
        test_gray(:, :, i) = rgb2gray(X(:, :, :, i));
        % 调用rgb2gray函数，把彩色图像变为灰色图像，降低通道数，3->1
    end

    test_gray = test_gray / 255;
    X_test = reshape(test_gray, 32 * 32, test_data_size);
    test_label_size = length(y);
    test_labels = zeros(10, test_label_size);
    % 0-9的数字编号

    for i = 1:test_label_size
        temp = y(i, 1);
        test_labels(temp, i) = 1;
    end
    
    % 保存最终的数据集格式 test_final_32x32.mat
    %          训练数据集 X_test
    %          数据的长度 test_data_size
    %          数据的标签 test_labels
    save test_final_32x32.mat X_test test_data_size test_labels
end

load train_32x32.mat

for i = 1:100
    subplot(10, 10, i);
    imshow(rgb2gray(X(:, :, :, i)));
end

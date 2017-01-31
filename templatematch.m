info = mha_read_header('C:\Users\rabbit\Documents\Github\imageprocessing\1_T1.mha');
image = mha_read_volume(info);
train_abnormal_image =gpuArray(image(20:150,20:200,1));
train_normal_image =gpuArray(image(20:150,20:200,1));
image_abnormality_index=[0];
image_normal_index=[0];
array=[0];
count_1 = 0;
count_2 = 0;
template1 =rgb2gray(imread('C:\Users\rabbit\Pictures\template.png'));
template2 = rgb2gray(imread('C:\Users\rabbit\Pictures\template2.png'));
template1 = gpuArray(template1);
template2 = gpuArray(template2);
for i = 61:120
img = (image(20:150,20 :200,i)); %ith image for searching
R2D = mat2gray(img);
[R2D8,map] = gray2ind(R2D,128); 
img = ind2gray(R2D8,map);
img = mat2gray(img);
img = gpuArray(img);

c1 = normxcorr2(template1,img);
c_cpu1 = gather(c1);
[ypeak1, xpeak1] = find(c_cpu1==max(c_cpu1(:)));
temp1 = c_cpu1(ypeak1, xpeak1);

c2 = normxcorr2(template2,img);
c_cpu2 = gather(c2);
[ypeak2, xpeak2] = find(c_cpu2==max(c_cpu2(:)));
temp2 = c_cpu2(ypeak2, xpeak2);


    if ((temp2)>(0.40)|| (temp1>0.44))
        count_1= count_1 +1;
        train_abnormal_image(:,:,count_1) = img;
    else
        count_2 = count_2 + 1;
        train_normal_image(:,:,count_2)= img;
    end
     
  
end
size_1 = size(train_abnormal_image);
size_1 = size_1(3);
size_2 = size(train_normal_image);
size_2 = size_2(3);
group = ones(size_1,1);
for i=1:size_2
    group = cat(1,group,2);   
end

for i=size_1+1: size_1+size_2
    train_abnormal_image(:,:,i) = train_normal_image(:,:,i-size_1);
end
for i=1:size_1+size_2
    images{i} = gather(train_abnormal_image(:,:,i));
end

trainData = zeros(size_1 + size_2,131*181);
for ii=1:size_1 + size_2
    images{ii} = reshape(images{ii}', 1, size(images{ii},1)*size(images{ii},2));
    trainData(ii,:) = images{ii};
end
SVMStruct = svmtrain (trainData,group);
inputImg = rgb2gray(imread('C:\Users\rabbit\Pictures\testimage2.png'));
inputImg = im2double(inputImg);
inputImg = inputImg(40:230,20:280);
inputImg = imresize(inputImg, [131 181]);
inputImg = reshape (inputImg', 1, size(inputImg,1)*size(inputImg,2));
result = svmclassify(SVMStruct, inputImg);
disp(result);

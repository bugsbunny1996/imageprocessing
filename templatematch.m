        info = mha_read_header('C:\Users\rabbit\Documents\Github\imageprocessing\volume_1.mha');
image = mha_read_volume(info);
train_abnormal_image =gpuArray(image(20:150,20:200,1));
train_normal_image =gpuArray(image(20:150,20:200,1));
arrayab=gpuArray(zeros(20,2,1));
arrayno =gpuArray(zeros(20,2,1));
temp= gpuArray(zeros(50,2));
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
        points = detectFASTFeatures(img); 
        points = points.selectStrongest(20);
        temp = points.Location;
        arrayab(:,:,count_1) = temp;
        
    else
        count_2 = count_2 + 1;
%         train_normal_image(:,:,count_2)= img;
        points = detectFASTFeatures(img); 
        points = points.selectStrongest(20);
        temp = points.Location;
        arrayno(:,:,count_2) = temp;
    end
    
end


    group = ones(count_1,1);
    for i=1:count_2
        group = cat(1,group,2);   
    end
    
  
     for i=count_1+1: count_1+ count_2
%          train_abnormal_image(:,:,i) = train_normal_image(:,:,i-size_1);
           arrayab(:,:,i) = arrayno(:,:,i-count_1);
     end
   
    for i=1:count_1+count_2
%         images{i} = gather(train_abnormal_image(:,:,i));
        images{i} = gather(arrayab(:,:,i));
    end
    
    trainData = zeros(count_1 + count_2, 2 * 20 );
    
    for ii=1:count_1 + count_2
          images{ii} = reshape(images{ii}', 1, size(images{ii},1)*size(images{ii},2));
          trainData(ii,:) = images{ii};
         
    end

SVMStruct = svmtrain(trainData,group);%'useGPU','yes' );

inputImg = (imread('C:\Users\rabbit\Documents\Github\imageprocessing\testimage1.png'));
if size(inputImg,3)== 3
    inputImg = rgb2gray(inputImg);
end

inputImg = im2double(inputImg);
inputImgpoints = detectFASTFeatures(inputImg);
inputImgpoints = inputImgpoints.selectStrongest(20);
inputImgpoints = inputImgpoints.Location;
imshow(inputImg); hold on;
points = detectFASTFeatures(inputImg);
if size(points,1)>= 20 
plot(points.selectStrongest(20));
inputImg = reshape (inputImgpoints', 1, size(inputImgpoints,1)*size(inputImgpoints,2));
result = svmclassify(SVMStruct, inputImg);
end

info = mha_read_header('C:\Users\rabbit\Pictures\volume_1.mha');
image = mha_read_volume(info);
template =rgb2gray(imread('C:\Users\rabbit\Pictures\template.png'));
template = gpuArray(template);
img = (image(20:150,20 :200,113));

R2D = mat2gray(img);
[R2D8,map] = gray2ind(R2D,128); 
img = ind2gray(R2D8,map);
img = mat2gray(pic);
imshow(pic);

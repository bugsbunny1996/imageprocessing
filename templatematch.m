info = mha_read_header('C:\Users\rabbit\Pictures\volume_1.mha');
image = mha_read_volume(info);
img = uint8(image(20:150,2:200,80)); %80th image for searching

template =rgb2gray(imread('C:\Users\rabbit\Pictures\template.jpeg'));
%template = imrotate(template,90);

%passing image and template into gpu
img = gpuArray(img);
template = gpuArray(template);

c = normxcorr2(template,img);
[ypeak, xpeak] = find(c==max(c(:)));
yoffSet = ypeak-size(template,1);
xoffSet = xpeak-size(template,2);
hFig = figure;
hAx  = axes;
imshow(img,'Parent', hAx);
imrect(hAx, [gather(xoffSet+1), gather(yoffSet+1), size(template,2), size(template,1)]);

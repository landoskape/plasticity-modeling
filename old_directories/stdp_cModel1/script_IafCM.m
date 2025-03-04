

[iaf,~,~,trackWeights] = runo2_iafCM(1,1,1); 
tWeights = reshape(trackWeights,1000,size(trackWeights,3));

%%
figure(1); clf; 

subplot(2,2,[1 2]);
imagesc(iaf.ampaWeights); 

subplot(2,2,3); 
imagesc(trackWeights(:,:,end/2));

subplot(2,2,4); 
imagesc(trackWeights(:,:,end));









































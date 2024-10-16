





NO = 3; 
NP = 7; 


T = 10000; 
inputActive = randi(NO, T, NP) + NO*(0:(NP-1));
idxOffset = NO*(0:(NP-1));
oriCorr = 0.5;

for t = 1:T
    if rand()<oriCorr
        % Make a row of three
        idxStart = randi(NP-3);
        inputActive(t,idxStart:idxStart+3) = 1 + idxOffset(idxStart:idxStart+3);
    end
end


iOn = zeros(T,NO*NP);
for t = 1:T
    iOn(t,inputActive(t,:)) = 1;
end

idxOrder = [1:3:NO*NP, 2:3:NO*NP, 3:3:NO*NP];

figure(1); clf;
subplot(1,2,1); 
imagesc(iOn);

subplot(1,2,2);
imagesc(cov(iOn(:,idxOrder)))
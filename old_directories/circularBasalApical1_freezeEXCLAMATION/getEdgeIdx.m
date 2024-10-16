function [edgeIdx,startPos] = getEdgeIdx(ori,L,eL,zDim)
    if rem(L,2)~=1, error('L has to be odd'); end
    if ~exist('zDim','var'), zDim = [1 1]; end
    startPos = randi(L); % start position index
    idxPos = mod((startPos:startPos+eL-1)-1,L)+1;
    switch ori
        case 1 % horizontal
            edgeIdx = sub2ind([L,L,zDim(1)],(L+1)/2*ones(1,eL),idxPos,zDim(2)*ones(1,eL));
        case 2 % diagonal from right corner
            rowIdx = L+1-idxPos;
            edgeIdx = sub2ind([L,L,zDim(1)],rowIdx,idxPos,zDim(2)*ones(1,eL));
        case 3 % vertical
            edgeIdx = sub2ind([L,L,zDim(1)],idxPos,(L+1)/2*ones(1,eL),zDim(2)*ones(1,eL));
        case 4 % diagonal from left corner
            edgeIdx = sub2ind([L,L,zDim(1)],idxPos,idxPos,zDim(2)*ones(1,eL));
    end
end
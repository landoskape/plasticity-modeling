function edge = drawEdge(orientation,L)
    if ~exist('L','var'), L = 3; end
    if rem(L,2)~=1, error('L has to be odd'); end
    edge = zeros(L);
    switch orientation
        case 1
            edge((L+1)/2,:) = 1;
        case 2
            edge(:,(L+1)/2) = 1;
        case 3
            edge(sub2ind([L,L],1:L,1:L)) = 1;
        case 4
            edge(sub2ind([L,L],1:L,L:-1:1)) = 1;
    end
end
function Es = p5gb( u1, u2 )
%P5GB  Five-point calibrated relative pose problem (Grobner basis).
%   Es = P5( u1, u2 ) computes the esential matrices E according to
%   Nister-PAMI2004 and Stewenius-PRS2006.
%
%   Input:
%     u1, u2 ..  3x5 matrices, five corresponding points in
%     HOMOGENEOUS coordinates.
%
%   Output:
%     Es .. cell array of possible essential matrices.

% (c) 2007-05-10 Martin Matousek
% Last change: $Date$
%              $Revision$

% Linear equations for the essential matrix (eqns 7-9).

qT = [u1(1,:) .* u2(1,:); u1(2,:) .* u2(1,:); u1(3,:) .* u2(1,:);
      u1(1,:) .* u2(2,:); u1(2,:) .* u2(2,:); u1(3,:) .* u2(2,:);
      u1(1,:) .* u2(3,:); u1(2,:) .* u2(3,:); u1(3,:) .* u2(3,:) ];

% span of null-space
[U,~,~] = svd(qT);
XYZW = U(:,6:9);

% the matrix 'A'
A = p5_matrixA( XYZW );

A1 = A(:, [1, 3, 4, 2, 5, 9, 7, 11, 14, 17] );
A2 = A(:, [6, 10, 8, 12, 15, 18, 13, 16, 19, 20] );

if( rcond(A1) < eps || rcond(A2) < eps)
  Es = [];
  return
end

A = A1 \ A2;
M = zeros( 10, 10 );
M(1:6,:) = -A([1 2 3 5 6 8], :);
M(7,1) = 1;
M(8,2) = 1;
M(9,4) = 1;
M(10,7) = 1;

[ V, D ] = eig( M );

ok = imag( diag( D ) ) == 0;

SOLS = V(7:9,ok) ./ V(10,ok);

Evec = XYZW(:,1:3) * SOLS + XYZW(:,4);
Evec = Evec ./ sqrt( sum( Evec.^2 ) );

n = size( Evec, 2 );
Es = cell( 1, n );
for i = 1:n
  Es{i} = reshape( Evec(:,i), 3, 3 )';
end

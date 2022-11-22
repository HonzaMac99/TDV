% DEMO_P5  Verification of the p5 algorithm
% (it works for every non-degenerated 5-tuple of correspondences)

% (c) 2010-10-19, Martin Matousek
% Last change: $Date::                            $
%              $Revision$


if( ~exist( 'p5gb', 'file' ) )
  error( 'Cannot find five-point estimator. Probably PATH is not set.' );
end

u1 = randn( 2, 5 )*10;
u2 = randn( 2, 5 );

u1p = [ u1; ones( 1, 5 ) ];
u2p = [ u2; ones( 1, 5 ) ];

Es = p5gb( u1p, u2p ); % cell array of essential matrices

fprintf( 'det(E)         max alg err\n' );

for i = 1:numel( Es )
  E = Es{i};

  alg_err = sum( u2p .* ( E * u1p ) );

  fprintf( '%15g %15g\n', det(E), max( abs( alg_err ) ) );
end

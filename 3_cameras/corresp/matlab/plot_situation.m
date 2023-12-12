function plot_situation( corresp, substate, enh, menh )
%PLOT_SITUATION  Draw correspondence situation scheme.

% (c) 2010-11-09 Martin Matousek
% Last change: $Date$
%              $Revision$

persistent coords

opt.tbl_textspec = ...
    { 'fontsize', 12, 'fontname', 'fixed', 'fontweight', 'bold', ...
      'backgroundcolor', 1 * [1 1 1], 'edgecolor', 'k', 'linewidth', 1 };

opt.tbl_textspec = ...
    { 'fontsize', 10, 'fontname', 'courier', ... %'fontweight', 'bold', ...
      'backgroundcolor', 1 * [1 1 1], 'edgecolor', 'k', 'linewidth', 1 };


%set( gcf, 'units', 'pixels', 'pos', [10   280   650   450] );

opt.tbl_wd = 0.2;
opt.tbl_ht = 0.22;
opt.pairs_top = 0.75;

opt.tables_in_figure = false;

if( opt.tables_in_figure )
  opt.textaxpos = [0.01 0.01 0.24 0.98];
  opt.graphaxpos = [0.26 0.01 0.73 .98];
else
  opt.graphaxpos = [0.01 0.01 0.98 .98];
  opt.pos = [862   372   567   455];
end

if isempty( corresp ) % plot_situation( [], coords )
  coords = substate;
  set( gcf, 'pos', opt.pos )
  return
end

beg = coords.beg;
x = coords.u;

n = length( x );

m = corresp.m;
px = corresp.Xu;
camsel = corresp.camsel;

numx = length( coords.x );
numcam = length( corresp.camsel );

% ------------------------------------------------------------------------------
% emphasizing
enh_xi = zeros( numcam, numx ); % (cam, xid)
enh_xu = cell( n, 1 ); % {cam}(inl)
enh_cam = zeros( numcam, 1 );

enh_u = []; % (cam,uid)
enh_x = zeros( numx, 1 ); % (xid)

enh_m = cell( n, n );
for i = 1:n
  enh_xu{i} = zeros( size( corresp.Xu{i}, 1 ), 1 );
  for j = 1:n
    enh_m{i,j} = zeros( size( m{i,j}, 1 ), 1 ); % redundant, both i,j and j,i
  end
end

if( exist( 'enh', 'var' ) && ~isempty( enh ) )
  for i = 1:size( enh, 1 )
    switch enh{i,1}
      case 'cam'
        enh_cam( enh{i,2} ) = 1;
      case 'x'
        enh_x( enh{i,2} ) = 1;
      case 'u'
        enh_u( enh{i,2}, enh{i,3} ) = enh{i,4}; %#ok
      case 'xi'
        enh_xi( enh{i,2}, enh{i,3} ) = 1;
      case 'x_u'
          enh_xu{ enh{i,2} }( enh{i,3} ) =  enh{i,4};
      case 'm'
        enh_m{enh{i,2}, enh{i,3}}(enh{i,4}) = enh{i,5};
        enh_m{enh{i,3}, enh{i,2}}(enh{i,4}) = enh{i,5};
      otherwise
        error '???'
    end

  end

end

if ~exist( 'menh', 'var' )
  menh = [];
end

% ------------------------------------------------------------------------------

clf
set( gcf, 'color', 'white' )

% ==============================================================================
% ==============================================================================
if( opt.tables_in_figure )
axes( 'pos', opt.textaxpos )
axis off

for i = 1:n
  tx = sprintf( '{{Xu%i}}', i );
  e = { 'edgecolor', 'black', 'linewidth', 1 };
  if( ~isempty( px{i} ) )
    if( camsel(i) )
      e = { 'edgecolor', 'red', 'linewidth', 1 };
    elseif( ~isempty( px{i} ) )
      e = { 'edgecolor', [0 0.5 0], 'linewidth', 1 };
    end

    if( enh_cam(i) )
      e = { 'edgecolor',[ 1 0.8 0], 'linewidth', 3 };
    end

    for j = 1:size( px{i}, 1 )
      q = px{i}(j,:);

      if( enh_xu{i}(j) < 0  )
        cq = '\color{gray}';
      elseif( enh_xu{i}(j) )
        cq = '\color{orange}\bf';
      elseif( corresp.Xu_verified{i}( j ) )
        cq = '\color{red}';
      else
        cq = '\color{darkgreen}';
      end

      tx = [ tx sprintf( '\n{%s%c\\mid%i}', cq, q(1) + 'A' - 1, q(2) ) ];%#ok
    end
  end

  text( opt.tbl_wd*(i-1), 1, tx, 'VerticalAlignment', 'top', ...
        'HorizontalAlignment', 'left', opt.tbl_textspec{:}, e{:} );
end

pair_clr = { '\bf\color{gray}' '\color{blue}' '\bf\color{orange}' };

for i = 1:n
  for j = i+1:n
    tx = sprintf( '{m%i%i}', i, j );
    for k=1:size( m{i,j}, 1 )
      tx = [ tx sprintf( '\n{%s%i\\mid%i}', ...
                         pair_clr{ enh_m{i,j}(k) + 2 }, ...
                         m{i,j}(k,1:2) ) ]; %#ok
    end

    if( ~isempty( menh ) && all( size( menh ) >= [i,j] ) ...
        && ~isempty( menh{i,j} ) )
      for k=1:size( menh{i,j}, 1 )
        tx = [ tx sprintf( '\n{%s%i\\mid%i}', ...
                           pair_clr{ 1 }, ...
                           menh{i,j}(k,1:2) ) ]; %#ok
      end
    end


    text( opt.tbl_wd*(j-2), opt.pairs_top-opt.tbl_ht*(i-1), tx, ...
          'VerticalAlignment', 'top', ...
          'HorizontalAlignment', 'left', opt.tbl_textspec{:} );
  end
end

% ------------------------------------------------------------------------------
else
  cr = sprintf( '\n' );
  tables_tx = [ '\xutables{%' cr ];
  for i = 1:n
    if( isempty( px{i} ) )
      tables_tx = [ tables_tx sprintf( '  \\xutableblack{%i}%%\n', i ) ]; %#ok
    else
      if( camsel(i) )
        tables_tx = [ tables_tx sprintf( '  \\xutablered{%i}{', i ) ]; %#ok
      else
        tables_tx = [ tables_tx sprintf( '  \\xutablegreen{%i}{', i ) ]; %#ok
      end

      for j = 1:size( px{i}, 1 )
        if( corresp.Xu_verified{i}( j ) )
          ptx = sprintf( '\\pairred{%c}{%i}', px{i}(j,:) + ['A'-1, 0 ] );
        else
          ptx = sprintf( '\\pairgreen{%c}{%i}', px{i}(j,:) + ['A'-1, 0 ] );
        end

        if( enh_xu{i}(j) < 0  )
          ptx = [ '\enhm{' ptx '}' ]; %#ok
        elseif( enh_xu{i}(j) )
          ptx = [ '\enhp{' ptx '}' ]; %#ok
        end

        tables_tx = [ tables_tx ptx ]; %#ok
      end
      tables_tx = [ tables_tx '}%' cr ]; %#ok
    end
  end
  tables_tx = [ tables_tx '}%' cr ];

  tables_tx = [ tables_tx '\mtables{%' cr ];
  for i = 1:(n-1)
    for j = 2:n
      if( j  <= i )
        tables_tx = [ tables_tx '\mtableskip' ]; %#ok
      else
        tables_tx = [ tables_tx sprintf( '\\mtable{%i}{%i}{', i, j ) ]; %#ok
        for k=1:size( m{i,j}, 1 )
          ptx = sprintf( '\\mpair{%i}{%i}', m{i,j}(k,1:2) );
          if( enh_m{i,j}(k) < 0 )
            ptx = [ '\enhm{' ptx '}' ]; %#ok
          elseif( enh_m{i,j}(k) > 0 )
            ptx = [ '\enhp{' ptx '}' ]; %#ok
          end

          tables_tx = [ tables_tx ptx ]; %#ok
        end

        if( ~isempty( menh ) && ...
            all( size( menh ) >= [i,j] ) ...
            && ~isempty( menh{i,j} ) )
          for k=1:size( menh{i,j}, 1 )
            ptx = sprintf( '\\enhm{\\mpair{%i}{%i}}', ...
                           menh{i,j}(k,1:2) );
            tables_tx = [ tables_tx ptx ]; %#ok
          end
        end
        tables_tx = [ tables_tx '}' cr ]; %#ok
      end
    end
    if( i < n-1 )
      tables_tx = [ tables_tx '\mtablesnextrow' cr ]; %#ok
    end
  end
  tables_tx = [ tables_tx '}' cr ];

end
% ==============================================================================
% ==============================================================================
ptsize = 18;

box_spec = [];
box_spec.blue =  { 'edgecolor', 'black', 'linewidth', 1 };
box_spec.red =   { 'edgecolor', 'red', 'linewidth', 2  };
box_spec.green = { 'edgecolor', [ 0 0.5 0], 'linewidth', 2};

pt_spec = [];
pt_spec.blue =  { 'color', 'blue', 'linewidth', 1  };
pt_spec.red =   { 'color', 'red', 'linewidth', 2  };
pt_spec.green = { 'color', [ 0 0.5 0], 'linewidth', 2};

xu_spec = [];
xu_spec.red =   { 'color', 'red', 'linewidth', 1  };
xu_spec.green = { 'color', [ 0 0.5 0], 'linewidth', 1};
xu_spec.blue = {};

axes( 'pos', opt.graphaxpos )
axis off
hold on

used_x = [];
for i = 1:n
  if( camsel(i) )
    state = 'red';
  elseif( ~isempty( px{i} ) )
    state = 'green';
  else
    state = 'blue';
  end

  h = patch( beg{i}(1) + [0 0 coords.imsz([1 1])], ...
             beg{i}(2) + [0 coords.imsz([2 2]) 0], ...
             [-2 -2 -2 -2], 'white', ...
             box_spec.(state){:} );

  if( enh_cam(i) )
    set( h, 'edgecolor',[ 1 0.8 0], 'linewidth', 4 );
  end
  x{i}(1,:) = x{i}(1,:) + beg{i}(1);
  x{i}(2,:) = x{i}(2,:) + beg{i}(2);

  h = text( beg{i}(1) + coords.tx{i,1}(1), ...
            beg{i}(2) + coords.tx{i,1}(2), 1, sprintf( 'img:%i', i ) );
  set( h, 'VerticalAlignment', coords.tx{i,3}, ...
          'HorizontalAlignment', coords.tx{i,2} );
  set( h, 'backgroundcolor', 0.9 * [1 1 1 ] )

  for j = 1:size( x{i}, 2 )
    h = plot3( x{i}(1,j), x{i}(2,j), x{i}(2,j) * 0, 'o', ...
               'markersize', ptsize, ...
               'markerfacecolor', 'w', pt_spec.blue{:} );
    f = [];
    if( ~isempty( px{i} ) )
      f = find( px{i}(:,2) == j );
    end

    if( any( size( enh_u ) < [i j ] ) )
      enh_u( i, j ) = 0; %#ok
    end

    if( enh_u( i, j ) )
      set( h, 'markerfacecolor', [1 1 0] );
    end


    id='';
    if( ~isempty( f ) )
      id = px{i}(f,1);

      if( any( enh_xi( i, id ) ) )
        set( h, 'markerfacecolor', [1 1 0] );
      end
      % if( corresp.Xu_verified{i}( j ) )
      %   set( h, pt_spec.red{:} );
      % else
      %   set( h, pt_spec.green{:} );
      % end
      id = sprintf( '%c', id+'A'-1);
    end
    h = text( x{i}(1,j), x{i}(2,j), 1, sprintf( '%s%i', id, j ) );
    set( h, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle' );
  end


  plot_cor( coords.x, x{i}, px{i}, enh_xu{i}, corresp.Xu_verified{i}, -1,...
            { 'linewidth', 1 })

  used_x( px{i}(:, 1) ) = 1; %#ok

end

for i = find( used_x )
  h = plot3( coords.x(1,i), coords.x(2,i), 0, 's', ...
             'markersize', ptsize, ...
             'markerfacecolor', 'w', pt_spec.red{:} );

  if( length( enh_x ) >= i && enh_x(i) )
    set( h, 'markerfacecolor', [1 1 0] );
  end


  id = sprintf( '%c', i+'A'-1);
  h = text( coords.x(1,i), coords.x(2,i), 1, id );
  set( h, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle' );
end

for i = 1:n
  for j = i+1:n
    if( ~isempty( m{i,j} ) )
      plot_cor( x{i}, x{j}, m{i,j}, enh_m{i,j} )
    end
    if( ~isempty( menh ) && all( size( menh ) >= [i,j] ) ...
        && ~isempty( menh{i,j} ) )
      plot_cor( x{i}, x{j}, menh{i,j}, ...
                -ones( 1, size( menh{i,j}, 1 ) ) );
    end
  end
end

hold off
axis equal
xlim( coords.xlim );
ylim( coords.ylim );
zlim( [-5 5] );

switch( corresp.state )
  case 'init'
    tx = 'init';
  case 'join'
    tx = [ 'join-' substate ];
  case 'newx'
    tx = [ 'newx-' num2str( corresp.statecounter ) '_' substate ];
  case 'verify'
    tx = [ 'verify-' num2str( corresp.statecounter ) '_' substate ];
  otherwise
    error
end

n = sum( corresp.camsel );
name = sprintf( 'situ%i-%s', n, tx );

if( ~opt.tables_in_figure )
  f = fopen( [ name '.tex' ], 'w' );
  fprintf( f, '%s', tables_tx );
  fclose(f);
end

drawnow
f = getframe( gcf );
imwrite( f.cdata, [ name '.png' ] );
%fig2eps( gcf, name, 'nohide' )

%input( 'Press Enter...' )

function plot_cor( pt1, pt2, m12, enh, state, depth, st )
%

if( nargin < 6 )
  state = [];
end

if( nargin < 7 )
  depth = -1;
end

if( nargin < 8 )
  depth = -1;
  st = {};
end

for i = 1:size( m12, 1 )
  i1 = m12( i, 1 );
  i2 = m12( i, 2 );

  if( numel( enh ) >= i && enh(i) > 0 )
    plot3( [ pt1( 1, i1 ) pt2( 1, i2 ) ], [ pt1( 2, i1 ) pt2( 2, i2 ) ], ...
           [ pt1( 1, i1 ) pt2( 1, i2 ) ] * 0 + depth, 'linewidth', 5, ...
           'color', [ 1 0.8 0] );
  end

  if( numel( enh ) >= i && enh(i) < 0 )
    plot3( [ pt1( 1, i1 ) pt2( 1, i2 ) ], [ pt1( 2, i1 ) pt2( 2, i2 ) ], ...
           [ pt1( 1, i1 ) pt2( 1, i2 ) ] * 0 + depth, 'linewidth', 5, ...
           'color', 0.7 * [ 1 1 1] );
  end


  if( isempty( state ) )
    spec = { 'color', 'b', 'linewidth', 1 };
  elseif( state(i) )
    spec = { 'color', 'r', 'linewidth', 1 };
  else
    spec = { 'color', [0 0.5 0], 'linewidth', 1 };
  end

  plot3( [ pt1( 1, i1 ) pt2( 1, i2 ) ], [ pt1( 2, i1 ) pt2( 2, i2 ) ], ...
         [ pt1( 1, i1 ) pt2( 1, i2 ) ] * 0 + depth, spec{:}, st{:} );

end

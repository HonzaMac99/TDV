%CORRESP_DEMO  Demonstration of the corresp package usage [script].

% (c) 2010-11-19 Martin Matousek
% Last change: $Date$
%              $Revision$

%% sample data
coords = struct( 'u', [], 'beg', [] );
coords.u{1} = [ 30 10; 30 50; 10 50; 40 30; 50 50; 20 35;10 10 ]';
coords.u{2} = [ 35 10; 10 10; 15 50; 50 10 ]';
coords.u{3}=  [ 50 20; 35 30; 10 50; 30 50; 50 10 ]';
coords.u{4} = [ 10 10; 50 10; 10 35; 50 50; 30 30 ]';
coords.u{5} = [ 50 10; ]';

coords.beg{1} = [ 145 40 ]';
coords.beg{2} = [ 125 170 ]';
coords.beg{3} = [ 20 0]';
coords.beg{4} = [ 250 0]';
coords.beg{5} = [ 240 140]';

coords.imsz = [60 60];
coords.xlim = [19 311];
coords.ylim = [-1 231];

coords.x = [ 90 150; 60 150; 80 100; 110 40; 75 110; 90 190 ]';
coords.tx = { [60 0], 'right', 'bottom';
              [60 60], 'right', 'top';
              [0 0], 'left', 'bottom';
              [60 30], 'right', 'middle';
              [0 0], 'left', 'bottom'; };

plot_situation( [], coords )

n = length(coords.u);

c = corresp( n );
c.verbose = 2;
c.add_pair( 1, 2, [ 4 1; 5 1; 3 3; 2 2] );
c.add_pair( 2, 3, [ 3 3; 3 4] );
c.add_pair( 1, 3, [ 1 1; 4 2; 6 1; 7 2] );
c.add_pair( 1, 4, [ 4 3; 4 4] );
c.add_pair( 2, 4, [ 1 3; 3 4;4 5] );
c.add_pair( 3, 4, [ 1 3; 5 5 ] );
c.add_pair( 1, 5, [ 1 1 ] );
%c.add_pair( 3, 5, [ 2 3 ] );

c.draw = @plot_situation;

%% run steps
plot_situation( c );
keyboard
c.start( 1, 2, [1 3] );

c.join_camera( 3, [1 2] );

c.new_x( 1, 3, 1 );

c.verify_x( 1, [] );

c.finalize_camera();

c.join_camera( 4, 3 );

c.new_x( 3, 4, 1 );

c.verify_x( 2, 3 );

c.finalize_camera();

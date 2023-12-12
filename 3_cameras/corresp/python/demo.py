# demo.py Demonstration of the corresp package usage [script]
#
# (c) 2020-11-11 Martin Matousek
# Last change: $Date$
#              $Revision$

import corresp
import numpy as np

c = corresp.Corresp( 5 )
c.verbose = 2
c.add_pair( 0, 1, np.array( [ [3,0], [4,0], [2,2], [1,1] ] ) )
c.add_pair( 1, 2, np.array( [ [2,2], [2,3] ] ) )
c.add_pair( 0, 2, np.array( [ [0,0], [3,1], [5,0], [6,1] ] ) )
c.add_pair( 0, 3, np.array( [ [3,2], [3,3] ] ) )
c.add_pair( 1, 3, np.array( [ [0,2], [2,3], [3,4] ] ) )
c.add_pair( 2, 3, np.array( [ [0,2], [4,4] ] ) )
c.add_pair( 0, 4, np.array( [ [0,0] ] ) )
#c.add_pair( 3, 5, np.array( [ [2, 3] ] ) )

c.start( 0, 1, np.array( [0, 2] ) )

c.join_camera( 2, np.array( [0, 1] ) )

c.new_x( 0, 2, np.array( [ 0 ] ) );

c.verify_x( 0, np.array( [], dtype=int ) )

c.finalize_camera()

c.join_camera( 3, np.array( [2] ) )

c.new_x( 2, 3, np.array( [0] ) )

c.verify_x( 1, np.array( [2] ) )

c.finalize_camera()

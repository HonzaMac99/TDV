syms uy vy ux vx f11 f12 f13 f21 f22 f23 f31 f32 f33
y = [uy; vy; 1];
x = [ux; vx; 1];
F = [f11 f12 f13; 
     f21 f22 f23; 
     f31 f32 f33];
 
e = transpose(y)*F*x;
de = [diff(e, ux);
     diff(e, vx);
     diff(e, uy);
     diff(e, vy)]



 
 
 
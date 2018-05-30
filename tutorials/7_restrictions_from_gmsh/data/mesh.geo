// This file is based on gmsh tutorial t5
// https://gitlab.onelab.info/gmsh/gmsh/blob/master/tutorial/t5.geo

lcar1 = .1;
lcar2 = .055;

Macro Outer
  Point(1) = {0,0,0,lcar1};
  Point(2) = {1,0,0,lcar1};
  Point(3) = {1,1,0,lcar1};
  Point(4) = {0,1,0,lcar1};
  Point(5) = {0,0,1,lcar1};
  Point(6) = {1,0,1,lcar1};
  Point(7) = {1,1,1,lcar1};
  Point(8) = {0,1,1,lcar1};

  Line(1) = {1,2};
  Line(2) = {2,3};
  Line(3) = {3,4};
  Line(4) = {4,1};
  Line(5) = {1,5};
  Line(6) = {2,6};
  Line(7) = {3,7};
  Line(8) = {4,8};
  Line(9) = {5,6};
  Line(10) = {6,7};
  Line(11) = {7,8};
  Line(12) = {8,5};

  Line Loop(13) = {1,2,3,4}; Plane Surface(14) = {13};
  Line Loop(15) = {1,6,-9,-5}; Plane Surface(16) = {15};
  Line Loop(17) = {-2,6,10,-7}; Plane Surface(18) = {17};
  Line Loop(19) = {-3,7,11,-8}; Plane Surface(20) = {19};
  Line Loop(21) = {4,5,-12,-8}; Plane Surface(22) = {21};
  Line Loop(23) = {9,10,11,12}; Plane Surface(24) = {23};
Return

Macro Inner
  x = 0.5; y = 0.5 ; z = 0.5; r = 0.3535;
  p1 = newp; Point(p1) = {x,  y,  z,  lcar2} ;
  p2 = newp; Point(p2) = {x+r,y,  z,  lcar2} ;
  p3 = newp; Point(p3) = {x,  y+r,z,  lcar2} ;
  p4 = newp; Point(p4) = {x,  y,  z+r,lcar2} ;
  p5 = newp; Point(p5) = {x-r,y,  z,  lcar2} ;
  p6 = newp; Point(p6) = {x,  y-r,z,  lcar2} ;
  p7 = newp; Point(p7) = {x,  y,  z-r,lcar2} ;

  c1 = newreg; Circle(c1) = {p2,p1,p7}; c2 = newreg; Circle(c2) = {p7,p1,p5};
  c3 = newreg; Circle(c3) = {p5,p1,p4}; c4 = newreg; Circle(c4) = {p4,p1,p2};
  c5 = newreg; Circle(c5) = {p2,p1,p3}; c6 = newreg; Circle(c6) = {p3,p1,p5};
  c7 = newreg; Circle(c7) = {p5,p1,p6}; c8 = newreg; Circle(c8) = {p6,p1,p2};
  c9 = newreg; Circle(c9) = {p7,p1,p3}; c10 = newreg; Circle(c10) = {p3,p1,p4};
  c11 = newreg; Circle(c11) = {p4,p1,p6}; c12 = newreg; Circle(c12) = {p6,p1,p7};

  l1 = newreg; Line Loop(l1) = {c5,c10,c4};    Surface(newreg) = {l1};
  l2 = newreg; Line Loop(l2) = {c9,-c5,c1};    Surface(newreg) = {l2};
  l3 = newreg; Line Loop(l3) = {c12,-c8,-c1};  Surface(newreg) = {l3};
  l4 = newreg; Line Loop(l4) = {c8,-c4,c11};   Surface(newreg) = {l4};
  l5 = newreg; Line Loop(l5) = {-c10,c6,c3};   Surface(newreg) = {l5};
  l6 = newreg; Line Loop(l6) = {-c11,-c3,c7};  Surface(newreg) = {l6};
  l7 = newreg; Line Loop(l7) = {-c2,-c7,-c12}; Surface(newreg) = {l7};
  l8 = newreg; Line Loop(l8) = {-c6,-c9,c2};   Surface(newreg) = {l8};
Return

Call Outer;
Call Inner;

outer_surface = newreg;
Surface Loop(outer_surface) = {14,16,18,20,22,24};

inner_surface = newreg;
Surface Loop(inner_surface) = {l8+1,l5+1,l1+1,l2+1,l3+1,l7+1,l6+1,l4+1}; Physical Surface(1) = {l8+1,l5+1,l1+1,l2+1,l3+1,l7+1,l6+1,l4+1};

inner_subdomain = newreg;
Volume(inner_subdomain) = inner_surface; Physical Volume(2) = inner_subdomain;

inner_complement_subdomain = newreg;
Volume(inner_complement_subdomain) = {outer_surface, inner_surface}; Physical Volume(1) = inner_complement_subdomain;

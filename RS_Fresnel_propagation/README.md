1.	Start with E0 = ones, z= 200, side = 10: Show diffraction pattern. Note that z = 20*side is almost far field.
2.	Change z from z=1 to z =1000. Show near and far field. Note border effect
3.	Change side down to side=1. Show free space propagator, which is calculated as D.
4.	Switch to kind=’phase’. Show phase of the (spherical) free space propagator. Switch to kernelFresnel and show the differences at the borders.
5.	Go back to kind=’abs’. E0 = exp (…) , z=200, side=30. Change kx from 0 to 0.5. Show that the plane wave is oblique.
6.	E0 = exp (…) , z=200, side=30, kx=0.3 : Switch to kind= ‘real’. Show oblique plane wave as a cosine. Spherical waves are diffraction from the border.

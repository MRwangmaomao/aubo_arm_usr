
import math
# translation: 
#   x: -0.0904424697245
#   y: 0.0827128670053
#   z: 0.0157324730325
# rotation: 
#   x: 0.0143232012768
#   y: 0.0770274858352
#   z: 0.996919784723
#   w: 0.00354332608545

w = 0.00354332608545
x = 0.0143232012768
y = 0.0770274858352
z = 0.996919784723

r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
p = math.asin(2*(w*y-z*x)) 
y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y)) 

# angleR = r*180/math.pi
# angleP= p*180/math.pi
# angleY = y*180/math.pi

# print (angleR)
# print (angleP)
# print (angleY) 

print (r)
print (p)
print (y) 
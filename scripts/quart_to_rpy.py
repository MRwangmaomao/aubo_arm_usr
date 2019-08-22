
import math


w = 0.0098606198788
x = 0.00471435414308
y = -0.0894849667007
z = -0.995928202119

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
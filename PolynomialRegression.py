import matplotlib.pyplot as plt

hours = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
speeds = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

# plt.scatter(hours, speeds)
# plt.xlabel('Hours')
# plt.ylabel('Speeds')
# plt.show()

# draw the line of Polynomial Regression
# import matplotlib.pyplot as plt
import numpy

mymodel = numpy.poly1d(numpy.polyfit(hours, speeds, 3))

myline = numpy.linspace(0, 22, 100)

plt.scatter(hours, speeds)
plt.plot(myline, mymodel(myline))
plt.show()


# Test R Square ( 0 means no relationship between hours and speed 1 means full relation)
from sklearn.metrics import r2_score

print(r2_score(speeds, mymodel(hours)))

# Now let's predict the speeds for any given hours like hours = 20

hour = 3
speed = int(mymodel(20))
print('If hour =', hour, ' Then speed will be = ', speed, ' Km/h')

# Sample output
# # If hour = 3  Then speed will be =  97  Km/h
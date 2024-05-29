import matplotlib.pyplot as plt 

plt.plotfile('C:\Users\pmzas2\Downloads\AmplitudeVectorNorm.dat', delimiter=' ', cols=(0, 1), 
             names=('col1', 'col2'), marker='o')
plt.show()
import qutip
import numpy as np

import matplotlib.pyplot as plt
pi=np.pi
b = qutip.Bloch()
b3d = qutip.Bloch3d()
b.make_sphere()
b.render()
#b.show()

x1 = np.random.uniform(pi / 2, 0, 20)
x2 = np.random.uniform(-pi / 2, 0, 20)
x3 = np.random.uniform(pi / 2, 0, 20)
th = np.linspace(0, 2*pi, 20)
temp=np.pi
xp = np.cos(x1)
yp = np.sin(x1)
zp = np.sin(x3)
x=np.cos(pi)
y=np.cos(pi)
z=np.cos(pi)

pnt = [0, 0, -1]
pnts = [xp, yp, zp]
b.add_points(pnts)

b.render()
b.show()
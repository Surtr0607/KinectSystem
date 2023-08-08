import matplotlib.pyplot as plt
import numpy as np
# x = list(range(1, 21))  # epoch array
# loss = [2 / (i**2) for i in x]  # loss values array
# plt.ion()
# for i in range(1, len(x)):
#     ix = x[:i]
#     iy = loss[:i]
#     plt.cla()
#     plt.title("loss")
#     plt.plot(ix, iy)
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.pause(0.5)
# plt.ioff()
# plt.show()

plt.ion()
# fig, ax = plt.subplots()
fig = plt.figure()
ax = plt.axes(projection='3d')
xx = np.random.random(20)*10-5
yy = np.random.random(20)*10-5
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))
ax.scatter(X,Y,Z)
plt.ioff()
plt.show()

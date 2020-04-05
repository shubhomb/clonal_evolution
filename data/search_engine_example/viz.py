import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 1000)
y = np.linspace(0, 5, 1000)

# these shouldn't change over game
search_engines = {
    (2.5, 4.8): "Google",
    (2.3, 2.0): "Ecosia",
    (4.0, 2.0): "Bing"
}


coords = list(search_engines.keys())
for k, v in search_engines.items():
    plt.annotate(xy=k, s=v)
    pass
x_list = [x for [x, y] in coords]
y_list = [y for [x, y] in coords]

xx, yy = np.meshgrid(x,y)
z = xx * yy


h = plt.contourf(x,y,z, levels=20)
plt.scatter(x_list, y_list, s=20, c='k' )

plt.xlabel("Speed")
plt.ylabel("Correctness")
plt.grid()
plt.title("environment function")
plt.colorbar(h)
plt.show()


flaw = lambda k, p, q: k * (np.exp(1-(xx-p)**2-(yy-q)**2))

h = plt.contourf(x,y,flaw(5, 2.3, 2.0) + z, levels=20)
plt.scatter(x_list, y_list, s=20, c='k' )

plt.xlabel("Speed")
plt.ylabel("Correctness")
plt.grid()
plt.title("responded fitness function")
plt.colorbar(h)
plt.show()

ks = [-1, 0, 1, 2.5, 5, 10]
f, axarr = plt.subplots(2, 3, sharex=True, sharey = True, figsize=(40,20))
for i in range(2):
    for j in range(3):
        axarr[i, j].contourf(x, y, flaw(ks[3 * i + j], 2.3, 2.0) + z, levels=50)
        axarr[i, j].set_xlabel("Speed", size=40)
        axarr[i, j].set_ylabel("Correctness", size=40)
        axarr[i, j].grid()
        axarr[i, j].scatter(x_list, y_list, s=50, c='k')
        for k, v in search_engines.items():
            axarr[i,j].annotate(xy=k, s=v,)
            pass
        axarr[i, j].set_title("responded fitness function k=%d"%ks[3 * i + j], size=40)
plt.savefig("response.png")
plt.show()


new_z = flaw(5, 2.3, 2.0) + z

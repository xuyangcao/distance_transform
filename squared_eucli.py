import numpy as np 
import matplotlib.pyplot as plt 

def eucli_dis(x, is_indicator=True):
    # init args
    f = x.copy()
    if is_indicator:
        f[f==0] = 1e10
        f[f==1] = 0
    k = 0
    v = np.zeros_like(f).astype(np.int16)
    z = np.zeros(len(f)+1)
    z[0] = -1e20
    z[1] = 1e20
    flag = 0

    # compute lower envelope
    for q in range(1, len(f)):
        flag = 0
        while flag != 1:
            #print('q={}, k={}, v[k]={}, z[k]={}'.format(q, k, v[k], z[k]))
            s = ((f[q] + q**2) - (f[v[k]] + v[k]**2)) / (2*q - 2*v[k])
            #print('s={}'.format(s))
            if s <= z[k]:
                k = k - 1
                flag = 0
            else:
                k = k + 1
                v[k] = q
                z[k] = s
                z[k + 1] = 1e20
                flag = 1
    # fill in values of Di_f(p)
    k = 0
    d = []
    for q in range(len(f)):
        while z[k+1] < q:
            k = k + 1
        #print('q={}, v[k]={}, f[v[k]]={}'.format(q, v[k], f[v[k]]))
        d.append((q - v[k])**2 + f[v[k]])
    d = np.array(d)

    return d

def one_dim_dis_demo():
    # gen data
    x = [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
    x = np.array(x, dtype=np.float32)

    # distance transform
    d = eucli_dis(x)

    # show results
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x)
    ax.set_title('origin data')
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(d)
    ax1.set_title('distance transform')
    fig.tight_layout()
    plt.show()

def two_dim_dis_demo():
    # gen image
    img = np.zeros((100, 100), dtype=np.float32)
    img[20:21, 40:50] = 1.
    img[60:61, 40:50] = 1.
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img, 'gray')
    ax.set_title('origin image')

    # distance transform
    w, h = img.shape
    for x in range(w):
        sample = img[x, :]
        img[x, :] = eucli_dis(sample, is_indicator=True)
    for x in range(h):
        sample = img[:, x]
        img[:, x] = eucli_dis(sample, is_indicator=False)

    # show results
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img, 'gray')
    ax.set_title('distance transform')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    one_dim_dis_demo()
    two_dim_dis_demo()

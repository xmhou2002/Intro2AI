import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from loadByWenLan import *

#catdog 108*2048 np.array

def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n)) #初始化为0
    idx = np.random.randint(0, m, k) #随机获得k个在[0,m)内的idx
    for i in range(k):
        centroids[i, :] = X[idx[i], :] #赋值给centroids
    return centroids

def cluster(X, centroids):
    m, n = X.shape
    k = centroids.shape[0]
    
    closestIdx = np.zeros(m)
    for i in range(m):
        minD = float('inf') 
        for j in range(k):
            cosDis = X[i,:].dot(centroids[j,:]) / np.linalg.norm(X[i,:]) * np.linalg.norm(centroids[j,:])
            if cosDis < minD:
                minD = cosDis
                closestIdx[i] = j
    return closestIdx

def update_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n)) #用X生成新centroids 必须加括号
    for i in range(k):
        indices = np.where(idx-i==0) #indices即i簇内点的idx行向量
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / #取上述indices对应的各行，按列求和，除簇中数据数
                           len(indices[0])).ravel() #最后展平 ravel():Return a contiguous flattened array.A copy is made only if needed.
    return centroids

def k_means(X, initCentroids, max_iters=100):
    m, n = X.shape
    k = initCentroids.shape[0]
    idx = np.zeros(m)
    centroids = initCentroids
    
    for i in range(max_iters):
        # m的维度是X.shape的行数也即样本数，find_closest_centroids函数返回的即数组，直接赋值给一开始初始化为0的idx数组即可
        idx = cluster(X, centroids)
        # centroids的类型为二维数组，即为compute_centroids函数计算得到的返回数组
        centroids = update_centroids(X, idx, k)
        
    return idx, centroids

initCentroids = init_centroids(catdog, 2)
idx, centroids = k_means(catdog, initCentroids)

# colors = np.array(["red", "magenta"])
pca = PCA(n_components=2)
# plt.plot(idx,color="red")
# plt.show()
catdogDim2=pca.fit_transform(catdog)
# print(catdogDim2)
# print(idx)
aData=catdogDim2[[x for x in range(len(catdog)) if idx[x]==0]]
bData=catdogDim2[[x for x in range(len(catdog)) if idx[x]==1]]

plt.scatter(aData[:,0],aData[:,1],marker='+')

plt.scatter(bData[:,0],bData[:,1],marker='o')

plt.show()

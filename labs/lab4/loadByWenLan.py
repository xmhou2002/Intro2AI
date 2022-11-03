import pickle

import numpy as np
import requests


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f, 0)
    f.close()
    return filename
def load_variable(filename):
    f = open(filename, 'rb')
    try: r = pickle.load(f)
    except: r = ""
    f.close()
    return r

url = "http://buling.wudaoai.cn/image_query"

cat = np.zeros((54, 2048))
dog = np.zeros((54, 2048))

# cat = load_variable('cat.txt')
# dog = load_variable('dog.txt')
# for i in range(1, 55):
#         catFile = {"image": open("./DogsCats/Cats/cat{}.jpeg".format(i), "rb")}
#         newCat = requests.post(url, files=catFile).json()['embedding']
#         cat[i-1,:] = np.array(newCat)
#         dogFile = {"image": open("./DogsCats/Dogs/dog{}.jpeg".format(i), "rb")}
#         newDog = requests.post(url, files=dogFile).json()['embedding']
#         dog[i-1,:] = np.array(newDog)
#     # dim=2048
# save_variable(cat,'cat.txt')
# save_variable(dog,'dog.txt')
# print(type(cat[0]))
# catdog = np.vstack((cat,dog))
# save_variable(catdog,'catdog.txt')

catdog = load_variable('catdog.txt')
# print(len(catdog))
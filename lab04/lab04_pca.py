import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def read_data():
    f = open('seoul_student.txt','r',encoding = "utf-8")    # 파일 열기
    lines = f.readlines()   # 데이터 세로 요소 길이 구하기 위해 모든 줄 읽는다
    dataArr = np.zeros((len(lines) - 1, 2), dtype= float)   # 데이터 저장할 배열
    f.seek(0)   # 파일 읽는 위치를 첫 줄로 이동
    f.readline()    # 첫 줄은 버린다

    index = 0
    while True:
        line = f.readline()
        if line == "":
            break

        dataArr[index] = line.split('\t')
        index += 1

    f.close()
    return dataArr

def normalization(dataArr):     # 정규화
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataArr)

def skl_PCA(dataArr):
    pca = PCA(n_components = 1)
    return pca.fit_transform(dataArr)

def get_mean(datas):
    return (sum(datas) / len(datas))


def get_covariance_matrix(dataArr): # 공분산 행렬 만드는 메소드

    dataArr[:,0] = dataArr[:,0] - get_mean(dataArr[:,0])
    dataArr[:,1] = dataArr[:,1] - get_mean(dataArr[:,1])

    dataArr_t = np.transpose(dataArr)

    return np.dot(dataArr_t, dataArr) / len(dataArr)  # 2*2 행렬 만들어짐


def e_sort(val, vec):
    index = np.argsort(val)[::-1]   # 고유값 내림차순 정렬 인덱스
    vec = vec[:,index]
    val = val[index]
    return vec

def pca(normal_Arr, dim = 1):
    cor_matrix = get_covariance_matrix(copy.deepcopy(normal_Arr))
    w, v = np.linalg.eig(cor_matrix)
    vec = e_sort(w, v)
    reduce_vec = vec[:,:dim] # 고유 벡터 축소
    return np.dot(normal_Arr, reduce_vec)

def draw_graph_2d(data):   # 데이터가 2차원일 경우 그래프 시각화
    plt.figure()
    plt.scatter(data[:,0], data[:,1], cmap = 'rainbow')
    plt.show()

def draw_graph_1d(data):   # 데이터가 1차원일 경우 그래프 시각화
    plt.figure()
    plt.scatter(data, [0] * len(data), cmap = 'rainbow')
    plt.show()

def main():
    dataArr = read_data()
    normal_Arr = normalization(dataArr)
    draw_graph_2d(normal_Arr)
    draw_graph_1d(skl_PCA(copy.deepcopy(normal_Arr)))
    draw_graph_1d(pca(copy.deepcopy(normal_Arr)))

if __name__ == '__main__':
    main()

import numpy as np
import random
import copy
from math import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from collections import OrderedDict

def read_data():
    f = open("C:/Users/ChoEunBin/Desktop/covid-19.txt",'r',encoding = "utf-8")     # 파일 열기
    lines = f.readlines()   # 데이터 세로 요소 길이 구하기 위해 모든 줄 읽는다
    dataArr = np.zeros((len(lines) - 1, 2), dtype= float)   # 데이터 저장할 배열
    f.seek(0)   # 파일 읽는 위치를 첫 줄로 이동
    f.readline()    # 첫 줄은 버린다

    index = 0
    while True:
        line = f.readline()
        if line == "":
            break

        tempList = line.split('\t')
        dataArr[index] = tempList[5:7]
        index += 1

    f.close()
    return dataArr

def normalization(dataArr):     # 정규화
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataArr)

def dbscan_func(dataArr):       # dbscan 밀도 기반 클러스터링
    dbscan = DBSCAN(eps = 0.1,min_samples = 2)
    return dbscan.fit_predict(dataArr)

def agglomerative_func(dataArr):    # 계층적 클러스터링
    agg = AgglomerativeClustering(n_clusters = 8, affinity = 'Euclidean', linkage = 'complete')
    return agg.fit_predict(dataArr)

def draw_graph(data, labels):   # 그래프 시각화
    plt.figure()
    plt.scatter(data[:,0], data[:,1], c = labels, cmap = 'rainbow')
    plt.show()

def euclidean_cal(list1, list2):    # 유클리디안 거리 계산
    return sqrt(sum(pow(a - b,2) for a, b in zip(list1, list2)))


class KMeans:
    def __init__(self, data, n):    # 객체 생성
        self.data = data
        self.n = n
        self.cluster = OrderedDict()

    def init_center(self):  # 초기 센터값 결정
        index = random.randint(0, self.n)
        index_list = []
        for i in range(self.n):
            while index in index_list:
                index = random.randint(0, self.n)
            index_list.append(index)
            self.cluster[i] = {'center': self.data[index], 'data':[]}

    def clustering(self, cluster):  # 가까운 센터값 그룹에 매핑
        for k in range(len(self.cluster)):  # 새로 클러스터링하기 위해 매핑되어 있던 데이터 비우기
            self.cluster[k]['data'] = []

        for i in range(len(self.data)):
            min_distance = 100    # 센터값, 데이터 사이의 최소 거리
            min_index = 100     # 최소거리인 데이터 인덱스
            min_key = 100       # 최소거리인 센터 인덱스
            for key, value in cluster.items():
                temp = euclidean_cal(value['center'], self.data[i])
                if min_distance > temp:
                   min_distance = temp
                   min_index = i
                   min_key = key

            self.cluster[min_key]['data'].append(self.data[min_index])  # 센터값 그룹에 매핑
        return self.cluster
            
    def update_center(self):    # 센터값 갱신
        for i in range(len(self.cluster)):
            self.cluster[i]['center'] = (sum(self.cluster[i]['data']))/len(self.cluster[i]['data'])

    def update(self):   # 센터값 갱신 및 클러스터링
        while True:
            state = False
            current_cluster = copy.deepcopy(self.cluster)
            self.update_center()
            self.clustering(self.cluster)
            # 기존 센터값과 클러스터링 후 센터값이 모두 동일할 경우 업데이트 종료
            for i in range(len(self.cluster)):
                if np.array_equal(current_cluster[i]['center'], self.cluster[i]['center']):
                    state = True
                else:
                    state = False
                    break
            if(state):
                break

    def fit(self):  # 외부에서 실행 호출하는 함수
        self.init_center()
        self.cluster = self.clustering(self.cluster)
        self.update()

        result, labels = self.get_result(self.cluster)
        draw_graph(result, labels)

    def get_result(self, cluster):  # 그래프 그리기 위헤 데이터 가공
        result = []
        labels = []
        for key, value in cluster.items():
            for item in value['data']:
                labels.append(key)
                result.append(item)
        return np.array(result), labels

def main():
    Arr = read_data()
    nomal_Arr = normalization(Arr)
    draw_graph(nomal_Arr,dbscan_func(nomal_Arr))
    draw_graph(nomal_Arr,agglomerative_func(nomal_Arr))
    KMeans_test = KMeans(nomal_Arr,8)
    KMeans_test.fit()

if __name__ == '__main__':
    main()
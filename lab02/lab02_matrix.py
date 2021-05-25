import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

def read_data():
    f = open("C:/Users/ChoEunBin/Desktop/seoul_tax.txt",'r',encoding = "utf-8")     # 파일 열기
    line = f.readline()     # 데이터 가로 요소 길이 구하기 위해 한 줄 읽는다
    lines = f.readlines()   # 데이터 세로 요소 길이 구하기 위해 첫 줄 이후 모든 줄 읽는다
    dataArr = np.zeros((len(lines), len(line.split())-1), dtype= int)   # 데이터 저장할 배열
    f.seek(0)   # 파일 읽는 위치를 첫 줄로 이동
    f.readline()    # 첫 줄은 버린다

    index = 0
    while True:
        line = f.readline()
        if line == "":
            break

        tempList = line.split()
        tempList = tempList[1:]
        tempArr = np.array(tempList, dtype= int)
        dataArr[index] = tempArr + dataArr[index]
        index += 1

    f.close()
    return dataArr


def cosine_main(dataArr):   # 모든 구의 코사인 거리 구하기
    cosine_Arr = np.zeros((len(dataArr), len(dataArr)), dtype= float)
    for i in range(len(cosine_Arr)):
        for k in range(len(cosine_Arr)):
            cosine_Arr[i][k] = cosine_distances(dataArr[[i]], dataArr[[k]])
    show_graph(cosine_Arr, 'cosine distance')


def manhattan_cal(list1, list2): # 맨하탄 거리 계산
    return (sum(abs(b-a) for a, b in zip(list1, list2)))

def manhattan_main(dataArr):    # 모든 구의 맨하탄 거리 구하기
    manhattan_Arr = np.zeros((len(dataArr), len(dataArr)), dtype= int)
    for i in range(len(manhattan_Arr)):
        for k in range(len(manhattan_Arr)):
            manhattan_Arr[i][k] = manhattan_cal(dataArr[i],dataArr[k])
    show_graph(manhattan_Arr, 'manhattan distance')

def euclidean_cal(list1, list2):    # 유클리디안 거리 계산
    return sqrt(sum(pow(a-b,2) for a, b in zip(list1, list2)))

def euclidean_main(dataArr):    # 모든 구의 유클리디안 거리 구하기
    euclidean_Arr = np.zeros((len(dataArr), len(dataArr)), dtype= float)
    for i in range(len(euclidean_Arr)):
        for k in range(len(euclidean_Arr)):
            euclidean_Arr[i][k] = euclidean_cal(dataArr[i],dataArr[k])
    show_graph(euclidean_Arr, 'euclidean distance')

def normalization(dataArr):     # 정규화
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataArr)

def show_graph(result_data, graph_title):    # 그래프 출력 함수
    plt.pcolor(result_data)
    plt.title(graph_title)
    plt.colorbar()
    plt.show()


def main():
    dataArr = read_data()
    print("1. 코사인 거리 그래프 (아무 키나 누르세요)")
    input()
    cosine_main(dataArr)
    print("2. 맨하탄 거리 그래프 (아무 키나 누르세요)")
    input()
    manhattan_main(dataArr)
    print("3. 유클리디안 거리 그래프 (아무 키나 누르세요)")
    input()
    euclidean_main(dataArr)
    normal_dataArr = normalization(dataArr)
    print("4. 정규화 코사인 거리 그래프 (아무 키나 누르세요)")
    input()
    cosine_main(normal_dataArr)
    print("5. 정규화 맨하탄 거리 그래프 (아무 키나 누르세요)")
    input()
    manhattan_main(normal_dataArr)
    print("6. 정규화 유클리디안 거리 그래프 (아무 키나 누르세요)")
    input()
    euclidean_main(normal_dataArr)

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt

f = open("C:/Users/ChoEunBin/Desktop/seoul.txt", 'r',encoding = "utf-8") # 파일 불러오기
f.readline()    # 한 줄 읽어서 인원 수가 저장된 줄로 넘어가기

resultArr = np.zeros((3,101), dtype = int)  # 결과 저장하는 배열
x = 0   # 줄 수 카운트하기 위한 변수

while True: # 파일이 끝날 때까지 한 줄씩 읽어들인다
    x += 1
    line = f.readline()
    if line == "":
        break

    if x == 1:  # x가 1이면 나이 별 '계' 인원수가 저장된 줄
        totalList = line.split()    # 탭을 기준으로 문자열을 나눠준다
        totalList = totalList[2:]   # 행정구역명, 성별 구분 단어 제외
        totalArr = np.array(totalList, dtype = int) # resultArr와의 연산을 위해 resultArr와 동일한 데이터 타입 지정해서 배열 생성
        resultArr[0] = totalArr + resultArr[0] # resultArr의 첫 번째 행에 더해 모든 행정 구역의 계 수치를 취합한다

    if x == 2:  # x가 2이면 나이 별 '남자' 인원수가 저장된 줄
        maleList = line.split()
        maleList = maleList[2:]
        maleArr = np.array(maleList, dtype = int)
        resultArr[1] = maleArr + resultArr[1]  # resultArr의 두 번째 행에 더해 모든 행정 구역의 남자 수치를 취합한다

    if x == 3:  # x가 3이면 나이 별 '여자' 인원수가 저장된 줄
        femaleList = line.split()
        femaleList = femaleList[2:]
        femaleArr = np.array(femaleList, dtype = int)
        resultArr[2] = femaleArr + resultArr[2]  # resultArr의 세 번째 행에 더해 모든 행정 구역의 여자 수치를 취합한다
        x=0    # 세 번째 줄까지 읽으면 x를 0으로 되돌려 다시 카운트

f.close()
print('계 : '," ".join(map(str, resultArr[0])))  # resultArr의 첫 번째 행 출력
print('계 총합 : ',np.sum(resultArr[0], axis = 0)) # 행의 모든 요소의 총합
print('계 평균 : ', np.mean(resultArr[0], axis = 0, dtype = int)) # 행의 모든 요소의 평균
print('계 분산 : ', int(np.var(resultArr[0], axis = 0)),'\n') # 행의 모든 요소의 분산


print('남자 : '," ".join(map(str, resultArr[1])))  # resultArr의 두 번째 행 출력
print('남자 총합 : ', np.sum(resultArr[1], axis = 0))
print('남자 평균 : ', np.mean(resultArr[1], axis = 0, dtype = int))
print('남자 분산 : ', int(np.var(resultArr[1], axis = 0)),'\n')

print('여자 : '," ".join(map(str, resultArr[2])))  # resultArr의 세 번째 행 출력
print('여자 총합 : ', np.sum(resultArr[2], axis = 0))
print('여자 평균 : ', np.mean(resultArr[2], axis = 0, dtype = int))
print('여자 분산 : ', int(np.var(resultArr[2], axis = 0)),'\n')

plt.figure(figsize=(16,4))  # 최초 창 크기 설정

plt.subplot(131)    # 하나의 창에 그래프 여러 개 그릴 수 있도록 구역 지정
plt.bar(np.arange(101),resultArr[0])
plt.title('Total')

plt.subplot(132)
plt.bar(np.arange(101),resultArr[1])
plt.title('Male')

plt.subplot(133)
plt.bar(np.arange(101),resultArr[2])
plt.title('Female')

plt.show()


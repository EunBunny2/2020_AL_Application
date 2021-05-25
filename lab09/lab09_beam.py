import numpy as np
from numpy import argmax

def greedy_search():
    score = 1   # score 1로 초기화
    sequences = list()
    for i in range(10):
        np.random.seed(int(score * 10))
        data = np.random.rand(5)    # 랜덤값 생성
        index = argmax(data)       # 랜덤값 중 가장 큰 값의 인덱스를 반환
        score = score * data[index]     # 가장 큰 값과 score을 곱해서 score 계산
        sequences.append(index)     # 가장 큰 값의 인덱스를 sequences에 추가
    sequences = [sequences, score]
    return sequences

def beam_search(k):
    score = 1   # score 1로 초기화
    sequences = [[list(), score]]
    for q in range(10):
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]   # 각 세트의 시퀀스와 score 분리
            np.random.seed(int(score * 10))
            data = np.random.rand(5)     # 각 세트의 score를 seed로 랜덤값 생성
            for j in range(5):
                candidate = [seq + [j], score * data[j]]    # 생성한 랜덤값의 score 계산
                all_candidates.append(candidate)    # score 세트를 all_candidates에 추가
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse = True)
        sequences = ordered[:k]      # all_candidate를 내림차순으로 정렬해 k개만 남기고 삭제
    return sequences


def main():
    print("greedy_search")
    print(greedy_search())
    print("\nbeam_search")
    beam_seq = beam_search(3)
    for i in beam_seq:
        print(i)

if __name__ == '__main__':
    main()
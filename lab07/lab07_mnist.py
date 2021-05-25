import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch import optim
import matplotlib.pyplot as plt

def read_data(train_path, test_path):

    # train 파일 읽어온다
    train = pd.read_csv(train_path)

    # 정답 항목인 label 데이터를 y_train에 저장
    y_train = train['label']

    # 전체 데이터 개수는 42000개이고 0~9의 숫자를 구분하므로
    # shape이 42000, 10이고 0으로 채워진 np 어레이를 생성
    y_list = np.zeros(shape=(y_train.size, 10))
    
    # One-Hot Encoding. y_train 값이 1이면 0 1 0 0 0 0 0 0 0 0
    for i, y in enumerate(y_train):
        y_list[i][y] = 1

    # y_train을 One-Hot Encoding 한 결과를 다시 y_train에 저장
    y_train = y_list

    # train에서 label 삭제
    del train['label']

    # 0~255 범위의 데이터 값들이 0~1 사이에 존재하도록 스케일링
    x_train = train.to_numpy() / 255

    # test 파일 읽어온다
    test = pd.read_csv(test_path)

    x_test = test.to_numpy() / 255

    return x_train, y_train, x_test

 # network model 정의
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # 784 차원 --> 512 차원
        self.fc2 = nn.Linear(512, 512)  # 512 차원 --> 512 차원
        self.fc3 = nn.Linear(512, 10)   # 512 차원 --> 10 차원

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3

def get_acc(pred, answer):  # 예측값과 정답을 받아서 정확도 측정
    correct = 0
    for p, a in zip(pred, answer):
        pv, pi = p.max(0)
        av, ai = a.max(0)
        if pi == ai:
            correct += 1
    return correct / len(pred)

def train(x_train, y_train, batch, lr, epoch):

    # model 생성
    model = MNISTModel()

    # model을 훈련시킬 때 model.train() 사용
    model.train()

    # 평균 제곱 오차(MSE)를 loss_function으로 사용
    # reduction="mean" 옵션은 모든 에러값들의 평균 도출
    loss_function = nn.MSELoss(reduction="mean")

    # Adam 알고리즘으로 모델을 최적화, 파라미터 업데이트.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # data 처리. 텐서 자료형으로 변환.
    x = torch.from_numpy(x_train).float()
    y = torch.from_numpy(y_train).float()

    # data를 batch 크기로 분리
    data_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch, shuffle = True)

    epoch_loss =[]  # loss 리스트
    epoch_acc = []  # accuracy 리스트

    for e in range(epoch):
        total_loss = 0
        total_acc = 0
        for data in data_loader:
            x_data, y_data = data

            # x_data를 model에 넣어서 결과 예측
            pred = model(x_data)

            # 예측한 결과와 정답 간의 차이를 구한다
            loss = loss_function(pred, y_data)

            # 이전의 학습 결과를 리셋
            optimizer.zero_grad()

            # 학습
            loss.backward()

            # update. 학습된 결과를 반영
            optimizer.step()

            total_loss += loss.item()   # loss 값들을 더해준다
            total_acc += get_acc(pred, y_data)  # accuracy 값들을 더해준다

        epoch_loss.append(total_loss / len(data_loader))    # 1 epoch이 끝나면 리스트에 loss 저장
        epoch_acc.append(total_acc / len(data_loader))    # 1 epoch이 끝나면 리스트에 accuracy 저장
        print("Epoch [%d] Loss: %.3f\tAcc: %.3f"% (e+1, epoch_loss[e], epoch_acc[e]))

    return model, epoch_loss, epoch_acc


def test(model, x_test, batch): # model로 test 해본다

    # model을 테스트할 때 model.eval() 사용
    model.eval()

    # data 처리. 텐서 자료형으로 변환.
    x = torch.from_numpy(x_test).float()

    # data를 batch 크기로 분리
    data_loader = torch.utils.data.DataLoader(x, batch, shuffle=False)

    preds = []
    for data in data_loader:    # epoch 없이 data_loader 돌려서 결과 예측
        pred = model(data)
        for p in pred:
            pv, pi = p.max(0)   # p는 One-Hot Encoding 되어 있는 데이터
            preds.append(pi.item()) # p의 max 값을 찾아서 preds에 저장

    return preds


def draw_graph(data):
    plt.plot(data)
    plt.show()

def save_pred(save_path, preds):
    # 예측된 결과를 submission 형식에 맞게 작성
    submission = pd.read_csv('sample_submission.csv', index_col='ImageId')
    submission["Label"] = preds

    # 결과를 csv 파일로 저장
    submission.to_csv(save_path)


def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    save_path = 'my_submission.csv'

    batch = 128
    lr = 0.001  # learning rate
    epoch = 10

    x_train, y_train, x_test = read_data(train_path, test_path)
    model, epoch_loss, epoch_acc = train(x_train, y_train, batch, lr, epoch)
    preds = test(model, x_test, batch)
    
    # 저장
    save_pred(save_path, preds)

    draw_graph(epoch_loss)
    draw_graph(epoch_acc)

if __name__ == '__main__':
    main()
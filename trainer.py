import os
import torch.optim
from tools import *
import time
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from model.vit import ViT
import scipy.io as scio
from dataloader import BtchLoadFftData, split_train_valid
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(3407)
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self,
                 learn_rate,
                 batch_size,
                 data_name
                 ):

        super(Trainer, self).__init__()
        self.class_number = 10
        self.batch_size = batch_size
        self.clip = -1
        self.lr = learn_rate
        self.data_name = data_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.NLLLoss().to(self.device)
        self.model = ViT()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model = self.model.to(self.device)

    def train(self, epoch, train_loader, train_len):

        self.model.train()
        correct = 0
        train_loss, train_acc, = 0, 0
        for batch_idx, batch in enumerate(train_loader):

            batch = [i.to(self.device) for i in batch]
            array, target = batch[0], batch[1]
            output, fe = self.model(array)

            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            if self.clip > 0:  # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            loss.backward()
            self.optimizer.step()
            train_loss += loss * target.size(0)
            argmax = torch.argmax(output, 1)
            train_acc += (argmax == target).sum()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.batch_size,
                    train_len, 100. * batch_idx * self.batch_size / train_len,
                    loss.item()))
        train_loss = torch.true_divide(train_loss, train_len)
        train_acc = torch.true_divide(train_acc, train_len)
        print('Train set: Average loss: {:.6f}, Accuracy: {}/{} ({:.5f}%)'.format(
            train_loss, correct, train_len, 100. * correct / train_len))
        return train_loss, train_acc

    def evaluate(self, epoch, epochs, test_loader, test_len, data_name):
        global best_acc
        # self.model.eval()
        correct_test, test_acc = 0, 0
        with torch.no_grad():
            tar, argm = [], []
            out = np.zeros(shape=(1, self.class_number))
            fes = np.zeros(shape=(1, 128))
            for test_idx, test_batch in enumerate(test_loader):
                batch = [i.to(self.device) for i in test_batch]
                array, target = batch[0], batch[1]
                output, fe = self.model(array)
                argmax = torch.argmax(output, 1)
                test_acc += (argmax == target).sum()
                pred_test = output.data.max(1, keepdim=True)[1]
                correct_test += pred_test.eq(target.data.view_as(pred_test)).cpu().sum()
                out = np.vstack(tup=(out, output.cpu().numpy()))
                fes = np.vstack(tup=(fes, fe.cpu().numpy()))
                torch.cuda.empty_cache()

                tar.extend(target.cpu().numpy())
                argm.extend(argmax.cpu().numpy())

            test_acc = torch.true_divide(test_acc, test_len)

            if epoch == 1:
                visualization(output=fes[1:],
                              y_predict=np.argmax(out[1:], axis=1),
                              labels_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                              data_name=data_name,
                              state="start")

            if epoch == epochs - 1:
                visualization(output=fes[1:],
                              y_predict=np.argmax(out[1:], axis=1),
                              labels_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                              data_name=data_name,
                              state="end")

            print('\nvalid set: Accuracy: {}/{} ({:.5f}%), Best_Accuracy({:.5f})'.format(
                correct_test, test_len, 100. * correct_test / test_len, best_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                print('The effect becomes better and the parameters are saved .......')

                p = precision_score(tar, argm, average='macro')
                recall = recall_score(tar, argm, average='macro')
                f1 = f1_score(tar, argm, average='macro')

                plot_confusion_matrix(y_true=tar, y_pred=argm,
                                      savename=f"result/{data_name}/Confusion-Matrix.png",
                                      title=f"Confusion-Matrix",
                                      classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

                plot_confusion_matrix_sim(y_true=tar, y_pred=argm,
                                          savename=f"result/{data_name}/Confusion-Matrix-sim.png",
                                          title=f"Confusion-Matrix-sim",
                                          classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

                result_text = f'result/{data_name}/{data_name}_log.txt'
                file_handle = open(result_text, mode='a+')
                file_handle.write('epoch:{},test_acc:{}, p:{}, recall:{},f1_score:{}\n'.format(
                    epoch, best_acc, p, recall, f1
                ))
                file_handle.close()
            return test_acc


def mian(learn_rate, batch_size, epochs, data_name):
    if torch.cuda.is_available():
        print(f'use gpu {torch.cuda.get_device_name()}')
    if not os.path.exists(f"result/{data_name}"):
        os.makedirs(f"result/{data_name}")

    T, V, F = split_train_valid()
    train_set = BtchLoadFftData(T, F)
    test_set = BtchLoadFftData(V, F)

    train_len, test_len = len(T), len(V)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    loss_train, acc_train, acc_test = [], [], []
    start = time.time()

    T = Trainer(learn_rate, batch_size, data_name)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = T.train(epoch=epoch,
                                        train_loader=train_loader,
                                        train_len=train_len)

        test_acc = T.evaluate(epoch=epoch,
                              epochs=epochs,
                              test_loader=test_loader,
                              test_len=test_len,
                              data_name=data_name)
        if torch.cuda.is_available():
            loss_train.append(train_loss.cuda().data.cpu().numpy())
            acc_train.append(train_acc.cuda().data.cpu().numpy())
            acc_test.append(test_acc.cuda().data.cpu().numpy())
        else:
            loss_train.append(train_loss.detach().numpy())
            acc_train.append(train_acc.detach().numpy())
            acc_test.append(test_acc.detach().numpy())

    end = time.time()
    train_time = end - start
    print("训练时间长度为  ==== > {} s".format(train_time))

    train_acc_(acc_train,
               save_name=f"result/{data_name}/train-acc",
               title=f"train-acc")

    train_loss_(loss_train,
                save_name=f"result/{data_name}/train-loss",
                title=f"train-loss")

    valid_acc_(acc_test,
               save_name=f"result/{data_name}/valid-acc",
               title=f"valid-acc")

    train_and_loss(acc_train, loss_train,
                   save_name=f"result/{data_name}/train-acc-and-loss",
                   title=f"train-acc-and-loss")

    # scio.savemat(f'mat_result/{data_name}.mat', {'train_acc': acc_train,
    #                                              'train_loss': loss_train,
    #                                              'acc_test': acc_test})


if __name__ == '__main__':
    best_acc = 0

    # "cwt", "stft"
    mian(learn_rate=0.001,
         batch_size=32,
         epochs=50,
         data_name="stft")

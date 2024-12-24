import torch
import random
import numpy as np
from itertools import cycle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import warnings
import matplotlib

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc("font", family='SimHei')


def visualization(output, y_predict, labels_list, data_name, state):
    data, predictions = output, y_predict
    tsne = PCA(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))

    colors = {
        0: 'b',
        1: 'g',
        2: 'r',
        3: 'c',
        4: 'm',
        6: 'y'
    }

    l = []
    for m, p in zip(reduced_data, predictions):
        p = int(p)
        if p in l:
            plt.scatter(m[0], m[1])
        else:
            plt.scatter(m[0], m[1], label=labels_list[p])
        l.append(p)

    plt.legend()
    plt.savefig(f'result/{data_name}/{state} vision.png', dpi=100)
    plt.close()


def train_loss_(loss, save_name, title):
    epoch = [i for i in range(len(loss))]
    plt.xticks(epoch[::5])
    plt.plot(epoch, loss, label="Train Loss")
    # plt.title('{}'.format(title))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(save_name))
    plt.close()


def train_acc_(acc, save_name, title):
    epoch = [i for i in range(len(acc))]
    plt.xticks(epoch[::5])
    plt.plot(epoch, acc, label="Train Acc")

    # plt.title('{}'.format(title))
    plt.ylabel('Acc')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(save_name))
    plt.close()


def valid_acc_(acc, save_name, title):
    epoch = [i for i in range(len(acc))]
    plt.xticks(epoch[::5])
    plt.plot(epoch, acc, label="Test Acc")

    # plt.title('{}'.format(title))
    plt.ylabel('Acc')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(save_name))
    plt.close()


def train_and_loss(acc, loss, save_name, title):
    epoch = [i for i in range(len(loss))]
    plt.xticks(epoch[::5])
    plt.plot(epoch, acc, "b", label="Train Acc")
    plt.plot(epoch, loss, "r", label="Train Loss")
    # plt.title('{}'.format(title))
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(save_name))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, savename, title, classes):  # 画混淆矩阵
    plt.figure(figsize=(15, 12), dpi=100)
    np.set_printoptions(precision=2)

    cm = confusion_matrix(y_true, y_pred)
    # 在混淆矩阵中每格的概率值
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.3f" % (c,), color='gray', fontsize=15, va='center', ha='center')
        # plt.text(x_val, y_val, c, color='red', fontsize=15, va='center', ha='center')

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    # plt.title(title, fontsize=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=-45, fontsize=15)
    plt.yticks(xlocations, classes, fontsize=15)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predict label', fontsize=15)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.close()


def plot_confusion_matrix_sim(y_true, y_pred, savename, title, classes):  # 画混淆矩阵
    plt.figure(figsize=(15, 12), dpi=100)
    np.set_printoptions(precision=2)

    cm = confusion_matrix(y_true, y_pred)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        # plt.text(x_val, y_val, "%0.3f" % (c,), color='gray', fontsize=15, va='center', ha='center')
        plt.text(x_val, y_val, c, color='red', fontsize=15, va='center', ha='center')

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    # plt.title(title, fontsize=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=-45, fontsize=15)
    plt.yticks(xlocations, classes, fontsize=15)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predict label', fontsize=15)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.close()

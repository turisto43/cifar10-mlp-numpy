import matplotlib.pyplot as plt


def draw_pic_loss(loss_all, save_path):
    plt.clf()
    plt.plot(loss_all)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.savefig(save_path)


def draw_pic_accuracy(accuracy_all, save_path):
    plt.clf()
    plt.plot(accuracy_all)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy')
    plt.savefig(save_path)
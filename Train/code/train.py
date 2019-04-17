import caffe
from caffe.proto import caffe_pb2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def main():

    caffe.set_device(0)
    caffe.set_mode_gpu()
    dir = '/home/ubuntu/Face-emotion-recognition-master/Train/code/'
    solver = load_solver(dir)
    loss, accuracy, test_interval = train(solver)
    plot_error(loss)
    plt.figure()
    plot_accuracy(accuracy, test_interval)
    plt.show()

def load_solver(dir):
    solver_proto = dir + 'FaceEmotionNet_solver.prototxt'
    solver = caffe.SGDSolver(solver_proto)
    return solver
def train(solver):
    niter = 76000
    test_interval = 1000
    train_loss = np.zeros(niter)
    test_acc = np.zeros(int(np.ceil(niter / test_interval)))
    display = 100
    for i in range(niter):
        solver.step(1)
        train_loss[i] = solver.net.blobs['Softmax_loss'].data
        if i % display == 0:
            print('Iteartion {}, Train loss: {}'.format(i,train_loss[i]))
        solver.test_nets[0].forward(start='conv1')

        if i % test_interval == 0:
            acc = solver.test_nets[0].blobs['accuracy'].data
            print('Iteration', i, 'testing...', 'accuracy:', acc)
            test_acc[i // test_interval] = acc

    accuracy = test_acc.tolist()
    sorted_accuracy = sorted(accuracy, reverse=True)
    print("The maximum test accuracy is {}".format(sorted_accuracy[0]))
    solver.net.save('/home/ubuntu/Face-emotion-recognition-master/Train/data/FaceEmotionNet.caffemodel')
    return train_loss, test_acc, test_interval

def plot_error(loss):
    plt.figure(1)
    iteration = np.arange(1, len(loss) + 1)
    plt.plot(iteration, loss, c='blue', alpha = 0.3)
    plt.xlabel('Number of Iteration')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Train loss vs iteration')




def plot_accuracy(accuracy, test_interval):
    plt.figure(2)
    plt.plot(test_interval*np.arange(len(accuracy)), accuracy, c='#56fca2')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Test accuracy vs iteration on FAR2013')





if __name__ == '__main__':
    main()




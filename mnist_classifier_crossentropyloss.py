import os

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

from common import train, test, save_state, save_data, draw_line_graph, draw_multi_lines_graph


# model
class MnistClassifierCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=(28 * 28), out_features=10)  # 784 -> 10

    def forward(self, x):
        x = x.view(-1, (28 * 28))  # flatten
        x = self.fc(x)
        return x


# experiment
def mnist_classifier_crossentropyloss():
    # paths
    path = dict()
    path['project'] = os.path.dirname(os.path.abspath(__file__))
    path['state'] = os.path.join(path['project'], 'epoch')
    path['dataset'] = os.path.join(path['project'], 'dataset')
    path['graph'] = os.path.join(path['project'], 'graph')
    path['array'] = os.path.join(path['project'], 'array')
    for key, value in path.items():
        if not os.path.exists(path[key]):
            os.mkdir(path[key])

    # parameters
    batch_size = 1000
    number_of_epochs = 20
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean = 0.1307
    std = 0.3081
    loss = nn.CrossEntropyLoss()
    info_per_batch = 6
    validation_ratio = 0.1

    # transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(mean,), std=(std,))
    ])

    # dataset
    train_dataset = torchvision.datasets.MNIST(root=path['dataset'], train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=path['dataset'], train=False, transform=transform, download=True)

    # validation dataset
    validation_limit = int((1 - validation_ratio) * len(train_dataset))
    index_list = list(range(len(train_dataset)))
    train_indexes, validation_indexes = index_list[:validation_limit], index_list[validation_limit:]
    train_sampler = SubsetRandomSampler(train_indexes)
    validation_sampler = SequentialSampler(validation_indexes)

    # dataset loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                    sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    # model
    model = MnistClassifierCrossEntropyLoss().to(device)

    # optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)

    epochs = np.arange(start=1, stop=(number_of_epochs + 1), step=1, dtype=int)

    print('Mnist Classifier CrossEntropyLoss')
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in epochs:
        info = 'Epoch {epoch_index}/{number_of_epochs}'
        print(info.format(epoch_index=epoch, number_of_epochs=number_of_epochs))

        # train
        train_loss, train_accuracy = train(model=model,
                                           device=device,
                                           loader=train_loader,
                                           optimizer=optimizer,
                                           loss=loss,
                                           info_per_batch=info_per_batch)
        info = 'Train: Average Loss: {train_loss:.5f}, Accuracy: % {train_accuracy:.2f}'
        print(info.format(train_loss=train_loss, train_accuracy=(100 * train_accuracy)))
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # validation
        validation_loss, validation_accuracy = test(model=model,
                                                    loader=validation_loader,
                                                    device=device,
                                                    loss=loss,
                                                    info_per_batch=info_per_batch,
                                                    info_name='Validation')
        info = 'Validation: Average Loss: {validation_loss:.5f}, Accuracy: % {validation_accuracy:.2f}'
        print(info.format(validation_loss=validation_loss, validation_accuracy=(100 * validation_accuracy)))
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        # test
        test_loss, test_accuracy = test(model=model,
                                        loader=test_loader,
                                        device=device,
                                        loss=loss,
                                        info_per_batch=info_per_batch,
                                        info_name='Test')
        info = 'Test: Average Loss: {test_loss:.5f}, Accuracy: % {test_accuracy:.2f}'
        print(info.format(test_loss=test_loss, test_accuracy=(100 * test_accuracy)))
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # epoch state
        state_file_name = 'mnist_classifier_crossentropyloss_epoch_{epoch_index}.pkl'.format(epoch_index=epoch)
        save_state(model=model, directory=path['state'], file_name=state_file_name)

    # train loss
    save_data(array=train_losses, directory=path['array'],
              file_name='mnist_classifier_crossentropyloss_train_loss.npy')
    draw_line_graph(x=epochs, y=train_losses,
                    x_label='Epoch', y_label='Loss',
                    title='Mnist Classifier CrossEntropyLoss Train Loss',
                    directory=path['graph'],
                    file_name='mnist_classifier_crossentropyloss_train_loss.png')

    # train accuracy
    save_data(array=train_accuracies, directory=path['array'],
              file_name='mnist_classifier_crossentropyloss_train_accuracy.npy')
    draw_line_graph(x=epochs, y=train_accuracies,
                    x_label='Epoch', y_label='Accuracy',
                    title='Mnist Classifier CrossEntropyLoss Train Accuracy',
                    directory=path['graph'],
                    file_name='mnist_classifier_crossentropyloss_train_accuracy.png')

    # validation loss
    save_data(array=validation_losses, directory=path['array'],
              file_name='mnist_classifier_crossentropyloss_validation_loss.npy')
    draw_line_graph(x=epochs, y=validation_losses,
                    x_label='Epoch', y_label='Loss',
                    title='Mnist Classifier CrossEntropyLoss Validation Loss',
                    directory=path['graph'],
                    file_name='mnist_classifier_crossentropyloss_validation_loss.png')

    # validation accuracy
    save_data(array=validation_accuracies, directory=path['array'],
              file_name='mnist_classifier_crossentropyloss_validation_accuracy.npy')
    draw_line_graph(x=epochs, y=validation_accuracies,
                    x_label='Epoch', y_label='Accuracy',
                    title='Mnist Classifier CrossEntropyLoss Validation Accuracy',
                    directory=path['graph'],
                    file_name='mnist_classifier_crossentropyloss_validation_accuracy.png')

    # test loss
    save_data(array=test_losses, directory=path['array'],
              file_name='mnist_classifier_crossentropyloss_test_loss.npy')
    draw_line_graph(x=epochs, y=test_losses,
                    x_label='Epoch', y_label='Loss',
                    title='Mnist Classifier CrossEntropyLoss Test Loss',
                    directory=path['graph'],
                    file_name='mnist_classifier_crossentropyloss_test_loss.png')

    # test accuracy
    save_data(array=test_accuracies, directory=path['array'],
              file_name='mnist_classifier_crossentropyloss_test_accuracy.npy')
    draw_line_graph(x=epochs, y=test_accuracies,
                    x_label='Epoch', y_label='Accuracy',
                    title='Mnist Classifier CrossEntropyLoss Test Accuracy',
                    directory=path['graph'],
                    file_name='mnist_classifier_crossentropyloss_test_accuracy.png')

    # loss
    draw_multi_lines_graph(
        lines=[
            dict(label='Train', data=dict(x=epochs, y=train_losses)),
            dict(label='Validation', data=dict(x=epochs, y=validation_losses)),
            dict(label='Test', data=dict(x=epochs, y=test_losses))
        ],
        x_label='Epoch', y_label='Loss',
        title='Mnist Classifier CrossEntropyLoss Loss',
        directory=path['graph'],
        file_name='mnist_classifier_crossentropyloss_loss.png')

    # accuracy
    draw_multi_lines_graph(
        lines=[
            dict(label='Train', data=dict(x=epochs, y=train_accuracies)),
            dict(label='Validation', data=dict(x=epochs, y=validation_accuracies)),
            dict(label='Test', data=dict(x=epochs, y=test_accuracies))
        ],
        x_label='Epoch', y_label='Accuracy',
        title='Mnist Classifier CrossEntropyLoss Accuracy',
        directory=path['graph'],
        file_name='mnist_classifier_crossentropyloss_accuracy.png')


# main
if __name__ == '__main__':
    mnist_classifier_crossentropyloss()

import os

import matplotlib.pyplot as graph
import numpy as np
import torch


def train(model, device, loader, optimizer, loss, one_hot_encoded=False, info_per_batch=10):
    model.train()
    number_of_batches = len(loader)
    batch_losses = []
    batch_accuracies = []
    for batch_index, (batch_input, batch_target) in enumerate(loader):
        batch_input, batch_target = batch_input.to(device), batch_target.to(device)
        optimizer.zero_grad()
        batch_output = model(batch_input)
        if one_hot_encoded:
            batch_target_one_hot_encoded = torch.nn.functional.one_hot(batch_target, 10).float()
            loss_calculation = loss(batch_output, batch_target_one_hot_encoded)
        else:
            loss_calculation = loss(batch_output, batch_target)
        loss_calculation.backward()
        optimizer.step()
        batch_loss = loss_calculation.item()
        batch_losses.append(batch_loss)
        batch_prediction = batch_output.max(dim=1, keepdim=True)[1]
        batch_correct = batch_prediction.eq(batch_target.view_as(batch_prediction)).sum().item()
        batch_size = len(batch_input)
        batch_accuracy = batch_correct / batch_size
        batch_accuracies.append(batch_accuracy)
        if (batch_index + 1) % info_per_batch == 0:
            info = 'Train: Batch {current_batch}/{number_of_batches}, Loss: {batch_loss:.5f}, Accuracy: % {batch_accuracy:.2f}'
            print(info.format(current_batch=(batch_index + 1), number_of_batches=number_of_batches,
                              batch_loss=batch_loss, batch_accuracy=(100 * batch_accuracy)))
    average_loss = sum(batch_losses) / number_of_batches
    accuracy = sum(batch_accuracies) / number_of_batches
    return average_loss, accuracy


def test(model, device, loader, loss, one_hot_encoded=False, info_name='Test', info_per_batch=10):
    model.eval()
    number_of_batches = len(loader)
    batch_loses = []
    batch_accuracies = []
    with torch.no_grad():
        for batch_index, (batch_input, batch_target) in enumerate(loader):
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            batch_output = model(batch_input)
            if one_hot_encoded:
                batch_target_one_hot_encoded = torch.nn.functional.one_hot(batch_target, 10).float()
                loss_calculation = loss(batch_output, batch_target_one_hot_encoded)
            else:
                loss_calculation = loss(batch_output, batch_target)
            batch_loss = loss_calculation.item()
            batch_loses.append(batch_loss)
            batch_prediction = batch_output.max(dim=1, keepdim=True)[1]
            batch_correct = batch_prediction.eq(batch_target.view_as(batch_prediction)).sum().item()
            batch_size = len(batch_input)
            batch_accuracy = batch_correct / batch_size
            batch_accuracies.append(batch_accuracy)
            if (batch_index + 1) % info_per_batch == 0:
                info = '{info_name}: Batch {current_batch}/{number_of_batches}, Loss: {batch_loss:.5f}, Accuracy: % {batch_accuracy:.2f}'
                print(info.format(current_batch=(batch_index + 1), number_of_batches=number_of_batches,
                                  batch_loss=batch_loss, batch_accuracy=(100 * batch_accuracy), info_name=info_name))
    average_loss = sum(batch_loses) / number_of_batches
    accuracy = sum(batch_accuracies) / number_of_batches
    return average_loss, accuracy


def save_state(model, directory, file_name):
    file_path = os.path.join(directory, file_name)
    state = model.state_dict()
    torch.save(obj=state, f=file_path)
    info = 'File: {file_name} is saved.'
    print(info.format(file_name=file_name))


def save_data(array, directory, file_name):
    file_path = os.path.join(directory, file_name)
    np.save(file=file_path, arr=array)
    info = 'File: {file_name} is saved.'
    print(info.format(file_name=file_name))


def load_data(directory, file_name):
    file_path = os.path.join(directory, file_name)
    array = []
    if os.path.exists(file_path):
        array = np.load(file_path)
        info = 'File: {file_name} is saved.'
        print(info.format(file_name=file_name))
    else:
        info = 'File: {file_name} does not exist.'
        print(info.format(file_name=file_name))
    return array


def draw_multi_lines_graph(lines, x_label, y_label, title, directory=None, file_name=None):
    graph.clf()
    labels = []
    for line in lines:
        label = line['label']
        labels.append(label)
        x = line['data']['x']
        y = line['data']['y']
        graph.xticks(x)
        graph.plot(x, y)
    graph.xlabel(xlabel=x_label)
    graph.ylabel(ylabel=y_label)
    graph.title(label=title)
    graph.legend(labels)
    if directory is not None:
        if file_name is None:
            file_name = '_'.join([word.lower() for word in title.split()]) + '.png'
        file_path = os.path.join(directory, file_name)
        graph.savefig(file_path)
        info = 'File: {file_name} is saved.'
        print(info.format(file_name=file_name))
    else:
        graph.show()


def draw_line_graph(x, y, x_label, y_label, title, directory=None, file_name=None):
    graph.clf()
    graph.xticks(x)
    graph.plot(x, y)
    graph.xlabel(xlabel=x_label)
    graph.ylabel(ylabel=y_label)
    graph.title(label=title)
    if directory is not None:
        if file_name is None:
            file_name = '_'.join([word.lower() for word in title.split()]) + '.png'
        file_path = os.path.join(directory, file_name)
        graph.savefig(file_path)
        info = 'File: {file_name} is saved.'
        print(info.format(file_name=file_name))
    else:
        graph.show()

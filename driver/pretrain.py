import os
import seaborn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import BERT, BertForSA
from module.optim import optim4GPU
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import config.hparams as hp
import sys


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = hp.lr, betas=(hp.adam_beta1, hp.adam_beta2), weight_decay: float = hp.adam_weight_decay, warmup_steps=hp.warmup_steps,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = hp.log_freq, args=None, global_step=0, path=None):
        """
        :param bert: MLM model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        self.args = args
        self.step = global_step
        self.path = path

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:1" if cuda_condition else "cpu")

        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        # self.model = BERTLM(bert, vocab_size).to(self.device)
        self.model = BertForSA(bert).to(self.device)

        # # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        total_steps = len(self.train_data) * hp.epochs
        self.optim = optim4GPU(self.model, total_steps)

        self.criterion = nn.CrossEntropyLoss()

        # Writer
        self.log_freq = log_freq
        # train
        self.train_loss_writer = SummaryWriter(f'{self.path.runs_path}/train/train_loss')
        self.train_attn_layer_writer = SummaryWriter(f'{self.path.runs_path}/train/attn_layer')

        self.num_params()


    def train(self):

        self.model.train()
        # try:
        for epoch in range(hp.epochs):

            # Setting the tqdm progress bar
            data_iter = tqdm.tqdm(enumerate(self.train_data),
                                  desc="EP_%s:%d" % ("train", epoch),
                                  total=len(self.train_data),
                                  bar_format="{l_bar}{r_bar}")

            running_loss = 0
            for i, data in data_iter:

                self.step += 1

                # data = {key: value.to(self.device) for key, value in data.items()}
                data["mlm_input"] = data["mlm_input"].to(self.device)
                data["input_position"] = data["input_position"].to(self.device)
                data["mlm_label"] = data["mlm_label"].to(self.device)

                logits = self.model.forward(data["mlm_input"], data["input_position"])

                self.optim.zero_grad()
                loss = self.criterion(logits, data["mlm_label"]).mean()

                loss.backward()
                self.optim.step()

                # loss
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                data_iter.set_description('Iter (loss=%5.3f)' % avg_loss)

                self.train_loss_writer.add_scalar('train_loss', loss, self.step)

                if self.step % hp.save_runs == 0:
                    for layer in range(hp.attn_layers):
                        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
                        print("Layer", layer + 1)
                        for h in range(hp.attn_heads):
                            self.draw(self.model.module.bert.layers[layer].multihead.attention[0][h].cpu().data,  # [0, h].data,  module
                                      [], [], ax=axs[h])
                        plt.savefig(f"{self.path.plt_train_attn_path}/Epoch{epoch}_train_step{self.step}_layer{layer + 1}")

                if self.step % hp.save_model == 0:
                    self.save_model(epoch, f"{self.path.bert_path}/bert")
                    self.save(self.step, f"{self.path.bert_path}")

                if hp.total_steps and hp.total_steps < self.step:
                    print('Epoch %d/%d : Average Loss %5.3f' % (epoch + 1, hp.epochs, running_loss / (i + 1)))
                    print('The Total Steps have been reached.')
                    self.save(self.step, f"{self.path.bert_path}")  # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f' % (epoch + 1, hp.epochs, running_loss / (i + 1)))

        self.save(self.step, f"{self.path.bert_path}")

        self.train_attn_layer_writer.close()
        self.train_loss_writer.close()


    def eval(self):
        self.model.eval()

        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")

        results = []
        with torch.no_grad():
            for i, data in data_iter:

                self.step += 1

                data = {key: value.to(self.device) for key, value in data.items()}

                logits = self.model.forward(data["mlm_input"], data["input_position"])

                accuracy, result = self.calculate(logits, data["mlm_label"])
                results.append(result)

                data_iter.set_description('Iter(acc=%5.3f)' % accuracy)

        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


    def calculate(self, logits, label_id):
        _, label_pred = logits.max(1)
        result = (label_pred == label_id).float()  # .cpu().numpy()
        accuracy = result.mean()
        return accuracy, result

    def stream(self, message):
        sys.stdout.write(f"\r{message}")

    def draw(self, data, x, y, ax):
        seaborn.heatmap(data,
                        xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,  # 取值0-1
                        cbar=False, ax=ax)

    def num_params(self, print_out=True):
        params_requires_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        params_requires_grad = sum([np.prod(p.size()) for p in params_requires_grad]) #/ 1_000_000

        parameters = sum([np.prod(p.size()) for p in self.model.parameters()]) #/ 1_000_000
        if print_out:
            print('Trainable total Parameters: %d' % parameters)
            print('Trainable requires_grad Parameters: %d' % params_requires_grad)


    def save_model(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def save_mlm_model(self, epoch, file_path="output/mlm_trained.model"):
        """
        Saving the current MLM model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def save(self, i, file_path):
        """ save current model """
        torch.save(self.model.state_dict(),  # save model object before nn.DataParallel
                   os.path.join(file_path, 'model_steps_' + str(i) + '.pt'))






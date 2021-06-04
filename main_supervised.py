import argparse
import torch
from supervised_data import SupDataset, get_train_valid_data
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from sklearn.metrics import f1_score
import time
from model import MiniTinySleepNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/ZhangHongjun/codes/sleep/openpai/data/sleepedf/sleep-cassette/eeg_fpz_cz",
                        help="where the sleep eeg data locate")

    parser.add_argument("--n_fold", type=int,
                        default=10,
                        help="training batch size")
    parser.add_argument("--st_fold", type=int,
                        default=0,
                        help="start fold")
    parser.add_argument("--en_fold", type=int,
                        default=9,
                        help="end fold")
    parser.add_argument("--n_epoch", type=int,
                        default=100,
                        help="train epoch")
    parser.add_argument("--cuda", type=int,
                        default=3,
                        help="fine tune pcg")
    parser.add_argument("--n_channel", type=int,
                        default="2",
                        help="number of channels")
    parser.add_argument("--optimizer", type=str,
                        default="sgd",
                        help="using sgd or adam")

    args = parser.parse_args()
    data_dir = args.data_dir
    n_fold = args.n_fold
    st_fold = args.st_fold
    en_fold = args.en_fold
    n_epoch = args.n_epoch
    cuda = args.cuda
    n_channel = args.n_channel
    opt_str = args.optimizer
    data_name = os.path.basename(data_dir)

    batch_size = 128
    lr = 0.001
    weight_decay = 0
    momentum = 0.9


    print("*" * 20, "parameters", "*" * 20)
    print("data ", data_dir)
    print("batch_size ", batch_size)
    print("lr ", lr)
    print("weight_decay ", weight_decay)
    print("optimizer", opt_str)


    print("*" * 20, "parameters", "*" * 20)

    device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    best_results = [{"epoch":0, "acc": 0} for f in range(st_fold, en_fold + 1)]
    for f in range(st_fold, en_fold + 1):
        clr_net = MiniTinySleepNet()
        train_files, valid_files = get_train_valid_data(data_dir=data_dir, n_fold=n_fold, fold_idx=f)

        # train_dataset = SupDataset(data_dir=data_dir, file_list=train_files, data_name=data_name)
        train_dataset = SupDataset(data_dir=data_dir, file_list=train_files, data_name=data_name)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


        valid_dataset = SupDataset(data_dir=data_dir, file_list=valid_files, data_name=data_name)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        if opt_str == "adam":
            optimizer = Adam(clr_net.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_str == "sgd":
            optimizer = torch.optim.SGD(clr_net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        # start training and validation
        print("start training...")
        # freeze the encoder
        clr_net = clr_net.to(device)
        lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.85)  # 随便设了两个参数, 不是最优的组合
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(1, n_epoch + 1):
            print('epoch ', epoch)
            start_time = time.time()
            clr_net.train()
            losses = []
            preds = []
            trues = []
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pred = clr_net(x)
                l = criterion(pred, y)
                l.backward()
                optimizer.step()

                prediction = torch.argmax(pred, 1)
                losses.append(l.detach().cpu().numpy())
                preds.append(prediction.cpu().numpy())
                trues.append(y.cpu().numpy())
            if len(losses) == 1:
                losses = losses[0]
                preds = preds[0]
                trues = trues[0]
            else:
                losses = np.array(losses)
                preds = np.concatenate(preds, axis=0)
                trues = np.concatenate(trues, axis=0)

            train_loss = np.mean(losses)
            train_acc = np.mean(preds == trues)
            train_f1 = f1_score(trues, preds, average="macro")
            print("train loss {:3f} acc {:3f} f1 score {:3f}".format(train_loss, train_acc, train_f1))

            # evaluation
            clr_net.eval()
            with torch.no_grad():
                losses = []
                preds = []
                trues = []
                for x, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = clr_net(x)
                    l = criterion(pred, y)
                    prediction = torch.argmax(pred, 1)
                    losses.append(l.detach().cpu().numpy())
                    preds.append(prediction.cpu().numpy())
                    trues.append(y.cpu().numpy())
                if len(losses) == 1:
                    losses = losses[0]
                    preds = preds[0]
                    trues = trues[0]
                else:
                    losses = np.array(losses)
                    preds = np.concatenate(preds, axis=0)
                    trues = np.concatenate(trues, axis=0)

                valid_loss = np.mean(losses)
                valid_acc = np.mean(preds == trues)
                valid_f1 = f1_score(trues, preds, average="macro")
                print("valid loss {:3f} acc {:3f} f1 score {:3f}".format(valid_loss, valid_acc, valid_f1))
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_results[f]["epoch"] = epoch
                    best_results[f]["acc"] = valid_acc
            lr_scheduler.step()

            end_time = time.time()
        print("epoch {}, {}s".format(epoch, end_time - start_time))

        # torch.save(clr_net.state_dict(), os.path.join(model_folder, 'fine_tune.pkl'))
        print("best_results")
        print(best_results)
        print("finish fine-tune")


if __name__ == '__main__':
    main()


# python3 main_supervised.py --cuda=3 --optimizer=sgd|tee supervised_sgd.txt
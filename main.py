import parser
import torch
from numpy import load
import data
import metrics
import models
import optimizers
import runner
import numpy as np
import pandas as pd



def main():
    # Add seed
    random_seed = 42
    torch.manual_seed(random_seed)
    args = parser.get()
    X_train = load('./datas/X_train.npy')
    y_train = load('./datas/y_train.npy')
    X_test = load('./datas/X_test.npy')
    train_dataset = data.DatasetXy(X_train, y_train)
    test_dataset = data.DatasetX(X_test)
    data_class = data.Dataloader(args, train_dataset, test_dataset)

    train, test = data_class.train(), data_class.test()    

    model = models.get(args)
    optimizer = optimizers.get(args, model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_metrics = runner.run(
            model,
            criterion,
            optimizer,
            train,
            True,
            {"loss": metrics.loss, "accuracy": metrics.accuracy},
        )
        metrics.print_metrics(train_metrics)
   
    y_test_pred = runner.run(
        model,
        criterion,
        optimizer,
        test,
        False,
        {"loss": metrics.loss, "accuracy": metrics.accuracy},
    )

    print(y_test_pred)
    y_test_pred = [item for sublist in y_test_pred for item in sublist]
    #print((y_test_pred[0]).shape)
    #_, y_pred = torch.max(y_test_pred, dim = 1)
    

    #y_pred = torch.round(y_test_pred)
    # _, y_pred = torch.max(y_test_pred, dim = 1)
    # y_pred = y_pred.cpu().numpy()
    #print(len(y_pred_list))
    #print(y_pred.type)
    y_test = np.asarray(y_test_pred)
    pd.DataFrame({"Id": np.arange(len(y_test)), "Category": y_test}).astype(int).to_csv(
    "solution.csv", index=False)

if __name__ == "__main__":
    main()

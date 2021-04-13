import torch


def save_model(epoch, model, optimizer, loss_func, lr_schedule, path):
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_func,
            'lr_schedule': lr_schedule.state_dict()
        }, path)
        return True
    except Exception as exp:
        print(f"Error occurs while saving model to {path}. Error info: {exp}")
        return False


def load_model(model, optimizer, lr_schedule, path):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])

        return dict(zip(["model", "optimizer", "lr_schedule", "epoch"], [model, optimizer, lr_schedule, epoch]))
    except Exception as exp:
        print(f"Error occurs while load model from {path}. Error info: {exp}")
        raise RuntimeError(f"Load Model from {path} failed! Error info: {exp}")
        # return dict(zip(["model", "optimizer", "lr_schedule", "epoch"], [None, None, None, None]))


def gen_model_save_name(model, dataset, epoch, **kwargs):
    sn = f"{model}_{dataset}_{epoch}"
    ks = '_'.join([f"{k}={v}" for k, v in kwargs.items()])
    sn = "_".join([sn, ks])
    return sn


if __name__ == "__main__":
    sn = gen_model_save_name("gen", "aids", 5, loss='cross', repeats=1000)
    print(sn)

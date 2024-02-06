# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, _LRScheduler, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.lr_scheduler import WarmUpLR

from vww_model import mobilenet_v1
from utils.dataloaders import get_dataloader
import torchsummary
from tqdm import tqdm, trange
from utils.utils import adjust_learning_rate, DescStr
from torchprofile import profile_macs
def main(args):
  
    model = mobilenet_v1()
    if torch.cuda.is_available():
        model = model.cuda()
    torchsummary.summary(model, (3, args.image_size, args.image_size))
    
    
    train_loader, val_loader = get_dataloader(dataset_dir=args.data_dir, 
                                            batch_size=args.batch_size, 
                                            image_size=args.image_size, 
                                            num_workers=args.workers, 
                                            shuffle=True)
    

    model = train_epochs(model, 
                        train_loader, 
                        val_loader, 
                        args.epochs, 
                        args.lr, 
                        args.log_dir, 
                        args.weight_decay, 
                        args.warmup_epochs)

    # Save model save_path in args
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), args.save_path, "vww_96.pth"))
    


def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    writer: SummaryWriter = None
):
    
    
    model.train()
    train_loss = 0.0  # cost function error
    train_acc = 0.0
    pbar = tqdm(train_dataloader, position=1, leave=False)
    for batch_index, (images, labels) in enumerate(pbar):
        
        if torch.cuda.is_available():
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            train_loss = (loss.item() + batch_index*train_loss) / (batch_index + 1)
            _, preds = outputs.max(1)
            train_acc = (preds.eq(labels).sum()/preds.shape[0] + batch_index*train_acc) / (batch_index + 1)
            
        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        # update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)
        writer.add_scalar("Lr", optimizer.param_groups[0]["lr"], n_iter)
        desc = f"Epoch={epoch}, Idx/Data= {batch_index}/{len(train_dataloader)}, " +\
                f"Acc={train_acc*100:.2f}%, loss={train_loss:.3f}, lr={optimizer.param_groups[0]['lr']:.5f}"
        pbar.set_description(desc=desc)


def eval_training(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_function: torch.nn.Module,
    epoch: int = 0,
    log_to_tensorboard: bool = True,
    writer: SummaryWriter = None,
):
    
    model.eval()

    test_loss = 0.0  # cost function error
    test_acc = 0.0  # accuracy
    
    with torch.no_grad():
        pbar = tqdm(dataloader, position=1, leave=False)
        for idx, (images, labels) in enumerate(pbar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            _, preds = outputs.max(1)
            
            test_loss = (loss.item() + idx*test_loss) / (idx + 1)
            test_acc = (preds.eq(labels).sum()/preds.shape[0] + idx*test_acc) / (idx + 1)
            
            pbar.set_description(desc=f"Epoch={epoch}, Batch_id={idx}/{len(dataloader)}, loss={test_loss:.4f}, Acc={test_acc*100:.2f}%")

    # print("Evaluating Network.....")
    # loss = test_loss
    # accuracy = test_acc
    

    # add information to tensorboard
    if log_to_tensorboard and writer:
        writer.add_scalar("Test/Average loss", test_loss, epoch)
        writer.add_scalar("Test/Accuracy", test_acc, epoch)

    return test_acc, test_loss


def train_epochs(model, 
                 train_loader, 
                 val_loader,
                 epoch_count,
                 learning_rate, 
                 log_dir,
                 weight_decay=1e-4,
                 learning_rate_decay=1e-4,
                 warmup_epochs=0):
    writer = SummaryWriter(log_dir=log_dir)
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    # scheduler = LambdaLR(
    #     optimizer=optimizer, lr_lambda=lambda epoch: learning_rate_decay**epoch
    # )
    # warmup_scheduler = None
    # if warmup_epochs:
    #     warmup_scheduler = WarmUpLR(optimizer=optimizer, total_iters= warmup_epochs)
    
    loss_function = torch.nn.CrossEntropyLoss()
    best_accuracy = 0.0
    tbar = trange(1, epoch_count+1, position=0)
    
    for epoch in tbar:
        
        adjust_learning_rate(optimizer, epoch, learning_rate)
            
        train_one_epoch(
            model=model,
            train_dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer)
        
        eval_acc, eval_loss = eval_training(model,
                                    val_loader,
                                    loss_function,
                                    epoch,
                                    log_to_tensorboard=False,
                                    writer=writer)
        
        if best_accuracy < eval_acc:
            weights_path = f"{log_dir}/best_{epoch}_{eval_acc:.4f}.pth"
            # print(f"saving weights file to {weights_path}")
            best_accuracy = eval_acc
            with torch.no_grad():
                torch.save(model.cpu().state_dict(), weights_path)
                model = model.cuda()
        tbar.set_description(desc=f"Epoch={epoch}, Eval_loss={eval_loss:.4f}, Eval_acc={eval_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.5f}")
        
        writer.flush()
    writer.close()

    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size. Default value is 32 according to TF training procedure.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs. Default value is 500 according to TF training procedure.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of epochs for LR linear warmup.",
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Initial learning rate. Default value is 1e-3 according to TF training procedure.",
    )
    parser.add_argument(
        "--lr-decay",
        default=1e-4,
        type=float,
        help="Initial learning rate. Default value is 1e-4 according to TF training procedure.",
    )
    parser.add_argument(
        "--data-dir",
        default="../../../training/visual_wake_words/vw_coco2014_96/",
        type=str,
        help="Path to dataset (will be downloaded).",
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of data loading processes."
    )
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--image-size", default=96, type=int, help="Input image size (square assumed)."
    )
    parser.add_argument(
        "--save-path", default="trained_models", type=str, help="Path to save model."
    )
    
    parser.add_argument("--log-dir", type=str, default="trained_models")
    args = parser.parse_args()
    
    main(args)

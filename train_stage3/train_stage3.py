import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from Loss_stage3 import Loss_stage3

def train_fuse(model_stage3, optimizer, data_loader_sa):
    device = 'cuda'
    for data_iter_step, img in enumerate(data_loader_sa):
        loss_ssim, loss_int = model_stage3(img.to(device))

        # ssim loss and intensity loss
        loss_fusion = 0.7 * loss_ssim + 0.3 * loss_int
        if data_iter_step % 10 == 0:
            print('data_iter:', data_iter_step, 'loss_ssim:', loss_ssim.item(), 'loss_int:', loss_int.item())
        optimizer.zero_grad()
        loss_fusion.backward()
        optimizer.step()
        torch.cuda.synchronize()


if __name__ == '__main__':
    # Configuring dataset paths
    path2COCO = ""
    path2COCO_CPSTN = ""

    # Configure the size of the training image
    train_size = (320, 240)

    # Device for training: 'cuda' or 'cpu'
    device = 'cuda'

    # COCO Dataset
    dataset = Dataset(path2COCO, path2COCO_CPSTN, train_size_w = train_size[0], train_size_h = train_size[1])

    dataset_sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dataset_sampler,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    

    model_stage3 = Loss_stage3()
    model_stage3.to(device)

    for name, param in model_stage3.named_parameters():
        if param.requires_grad:
            print(name)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_stage3.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    print(optimizer)

    model_stage3.train(True)

    epochs = 20
    print('Start training!')

    # Train SAF
    print('Train Fusion!')
    for epoch in range(0, epochs):
        print('current epoch:', epoch)
        train_fuse(model_stage3, optimizer, data_loader)
        scheduler.step()
        torch.save(model_stage3.state_dict(), './weights/fusion_{}epoch.ckpt'.format(epoch+1))






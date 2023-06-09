import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from Loss_stage2 import Loss_stage2

def train_csc(model_stage2, optimizer, data_loader_sa, stage1_weight_path, epoch):
    device = 'cuda'
    optimizer.zero_grad()
    # Read the required data set for registration
    for data_iter_step, (img_vi, img_ir, conf_gt, str_conf_gt) in enumerate(data_loader_sa):
        

        loss_0, loss_1 = model_stage2(img_vi.to(device), img_ir.to(device), conf_gt.to(device), str_conf_gt.to(device))
        loss = loss_0 + loss_1 * (1 + 0.03 * epoch)

        if data_iter_step % 10 == 0:
            print('data_iter:', data_iter_step, 'loss_0:', loss_0.item(), 'loss_1:', loss_1.item())

        loss.backward()
        optimizer.step()

        model_stage2.load_state_dict(torch.load(stage1_weight_path), strict=False)
        torch.cuda.synchronize()


def train_ssr(model_stage2, optimizer, data_loader_sa, csc_weight_wossr, epoch):
    device = 'cuda'
    # Read the required data set for registration
    for data_iter_step, (img_vi, img_ir, conf_gt, str_conf_gt) in enumerate(data_loader_sa):

        loss_0, loss_1 = model_stage2(img_vi.to(device), img_ir.to(device), conf_gt.to(device), str_conf_gt.to(device))

        loss = loss_0

        if data_iter_step % 10 == 0:
            print('data_iter:', data_iter_step, 'loss_0:', loss_0.item(), 'loss_1:', loss_1.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_stage2.load_state_dict(csc_weight_wossr, strict=False)
        torch.cuda.synchronize()


if __name__ == '__main__':
    # Configuring dataset paths
    path2IVS = ""
    path2IVS_CPSTN = ""
    path2IVS_Label = ""

    # Configure the size of the training image
    train_size = (320, 240)

    # Load the model weights obtained from the first stage of training
    stage1_weight_path = "./weights/stage1_14epoch.ckpt"
    
    # Device for training: 'cuda' or 'cpu'
    device = 'cuda'

    dataset = Dataset(path2IVS, path2IVS_CPSTN, path2IVS_Label, train_size_w = train_size[0], train_size_h = train_size[1])

    dataset_sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dataset_sampler,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    

    model_stage2 = Loss_stage2()
    model_stage2.to(device)
    model_stage2.load_state_dict(torch.load(stage1_weight_path), strict=False)
    
    for k ,v in model_stage2.named_parameters():

        if ('csc0' in k) or ('csc1' in k) or ('ssr' in k):

            v.requires_grad = True
        else:
            v.requires_grad = False
            
    for name, param in model_stage2.named_parameters():
        if param.requires_grad:
            print(name)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_stage2.parameters()), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    print(optimizer)

    model_stage2.train(True)

    epochs_train_csc = 2
    print('Start training!')

    # Train CSC
    print('Train CSC!')
    for epoch in range(0, epochs_train_csc):
        print('current epoch:', epoch)
        train_csc(model_stage2, optimizer, data_loader, stage1_weight_path, epoch)
        scheduler.step()
        torch.save(model_stage2.state_dict(), './weights/stage2_csc_{}epoch.ckpt'.format(epoch+1))

    # Train SSR
    epochs_train_ssr = 10
    print('Train SSR!')
    csc_weight_path = "./weights/stage2_csc_2epoch.ckpt"
    csc_weight = torch.load(csc_weight_path)
    csc_weight_wossr = {key: value for key, value in csc_weight.items() if 'ssr' not in key}

    for epoch in range(0, epochs_train_ssr):
        print('current epoch:', epoch)
        train_ssr(model_stage2, optimizer, data_loader, csc_weight_wossr, epoch)
        scheduler.step()
        torch.save(model_stage2.state_dict(), './weights/stage2_ssr_{}epoch.ckpt'.format(epoch+1))





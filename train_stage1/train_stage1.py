import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from teacher.model.matchformer import Matchformer
from teacher.config.defaultmf import default_cfg
from reg_dataset import RegDataset
from IVS_dataset import IVSDataset
from Loss_stage1 import Loss_stage1


def train_one_epoch(model_stage1, teacher, optimizer, data_loader_reg, data_loader_sa):
    device = 'cuda'
    dataloader_iterator = iter(data_loader_sa)

    # Read the required data set for registration
    for data_iter_step, (reg_img0, reg_img1, reg_conf_gt) in enumerate(data_loader_reg):

        # Read the required data set for semantic awareness
        try:
            (sa_img0, sa_conf_gt0, sa_img1, sa_conf_gt1) = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(data_loader_sa)
            (sa_img0, sa_conf_gt0, sa_img1, sa_conf_gt1) = next(dataloader_iterator)


        batch = {'image0': reg_img0.to(device), 'image1': reg_img1.to(device)}
        with torch.no_grad():
            teacher(batch)
        teacher_pred = batch['conf_matrix']
        teacher_pred = torch.clamp(teacher_pred, 1e-6, 1 - 1e-6)

        optimizer.zero_grad()
        loss_reg, loss_sa, student_pred = \
                model_stage1(reg_img0.to(device), reg_img1.to(device), reg_conf_gt.to(device),
                sa_img0.to(device), sa_img1.to(device), sa_conf_gt0.to(device), sa_conf_gt1.to(device))


        stu_loss = nn.KLDivLoss(reduction='batchmean')
        student_loss = stu_loss(torch.log(student_pred), teacher_pred)

        if data_iter_step % 10 == 0:
            print('data_iter:', data_iter_step, 'reg_loss:', loss_reg.item(), 'sa_loss:', loss_sa.item(),  'student_loss:',
                  student_loss.item())


        loss_stage1 = loss_reg+ loss_sa*0.4+ student_loss * 0.003
        loss_stage1.backward()

        optimizer.step()
        torch.cuda.synchronize()


if __name__ == '__main__':
    # Configuring dataset paths
    path2COCO = ""
    path2COCO_CPSTN = ""
    path2IVS = ""
    path2IVS_CPSTN = ""
    path2IVS_Label = ""
    MatchFormer_weight_path = ""

    # Configure the size of the training image
    train_size = (320, 240)

    # Device for training: 'cuda' or 'cpu'
    device = 'cuda'

    # dataset for training registration
    Reg_data = RegDataset(path2COCO, path2COCO_CPSTN, train_size_w = train_size[0], train_size_h = train_size[1])

    # dataset for training semantic awareness
    Sa_data = IVSDataset(path2IVS,
                         path2IVS_CPSTN,
                         path2IVS_Label, train_size_w = train_size[0], train_size_h = train_size[1])


    Reg_sampler = torch.utils.data.RandomSampler(Reg_data)
    Sa_sampler = torch.utils.data.RandomSampler(Sa_data)

    data_loader_reg = torch.utils.data.DataLoader(
        Reg_data,
        sampler=Reg_sampler,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    data_loader_sa = torch.utils.data.DataLoader(
        Sa_data,
        sampler=Sa_sampler,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    model_stage1 = Loss_stage1()
    model_stage1.to(device)

    for name, param in model_stage1.named_parameters():
        if param.requires_grad:
            print(name)


    # Knowledge distillation using MatchFormer
    use_registration_teacher = True
    if use_registration_teacher == True:
        teacher = Matchformer(config=default_cfg)

        # loading the weights of matchformer
        teacher.load_state_dict(torch.load(MatchFormer_weight_path), strict=False)
        teacher.eval().to(device)

    optimizer = torch.optim.Adam(model_stage1.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.908)
    print(optimizer)

    model_stage1.train(True)

    epochs = 13
    print('Start training!')
    for epoch in range(0, epochs):
        print('current epoch:', epoch)
        train_one_epoch(model_stage1, teacher, optimizer, data_loader_reg, data_loader_sa)
        scheduler.step()
        torch.save(model_stage1.backbone.state_dict(), './weights/stage1_{}epoch.ckpt'.format(epoch+1))



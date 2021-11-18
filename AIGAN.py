import gc
import torch
import torch.nn.functional as F
from torch import nn
from advertorch.attacks import LinfPGDAttack
## My functions
from models import Generator,Discriminator

## Functions
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adv_loss(probs_model, onehot_labels, is_targeted=False):   
    # C&W loss function
    real = torch.sum(onehot_labels * probs_model, dim=1)
    other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
    zeros = torch.zeros_like(other)
    if is_targeted:
        loss_adv = torch.sum(torch.max(other - real, zeros))
    else:
        loss_adv = torch.sum(torch.max(real - other, zeros))
    return loss_adv

class AIGAN():
    def __init__(self,device,target_model,img_channels):
        self.device = device
        self.lr = 10**(-3) ## MNIST
        self.th_perturb = 0.3
        self.box_min = 0
        self.box_max = 1
        self._use_attacker = True
        self.target_out_labels = 10
        self.epoch_of_change = 10
        ## Models
        self.target_model = target_model
        self.netG = Generator(img_channels).to(device)
        self.netG.apply(weights_init)
        self.netD = Discriminator(img_channels).to(device)
        self.netD.apply(weights_init)
        self.attacker = LinfPGDAttack(self.target_model,loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=self.th_perturb,nb_iter=40,eps_iter=0.01,rand_init=True,
                clip_min=self.box_min,clip_max=self.box_max,targeted=False)

        ## Optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=self.lr)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=self.lr)
        
    
    def train(self,train_dataloader,n_epochs):
        self.netG.train()
        self.netD.train()
        log_file = 'log.txt'
        for epoch in range(n_epochs):
            if epoch == self.epoch_of_change:
                self._use_attacker = False
            if epoch == 120:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=0.0001)
            if epoch == 200:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=0.00001)
            
            loss_D_sum = 0
            loss_S_D_sum = 0
            loss_adv_sum = 0
            loss_pert_sum = 0
            loss_S_G_sum = 0
            loss_perturb_sum = 0
            loss_adver_sum = 0
            loss_G_sum = 0
            fake_acc_sum = 0
            for i,data in enumerate(train_dataloader,start=0):
                gc.collect()
                ## Batchwise
                images,labels = data
                images,labels = images.to(self.device),labels.to(self.device)
                fake_acc,loss_D,loss_S_D,loss_adv,loss_pert,loss_G,loss_S_G,loss_perturb,loss_adver = \
                                                    self.train_batch(images,labels)
                loss_D_sum += loss_D
                loss_S_D_sum += loss_S_D
                loss_adv_sum += loss_adv
                loss_pert_sum += loss_pert
                loss_G_sum += loss_G
                loss_S_G_sum += loss_S_G
                loss_perturb_sum += loss_perturb
                loss_adver_sum += loss_adver
                fake_acc_sum += fake_acc
            n_bs = len(train_dataloader)
            out = "Epoch:{}/{}::Fake acc:{}, Loss_D_total:{}, Loss_S_D:{}, Loss_adv:{}, Loss_pert:{}".format(
                epoch+1,n_epochs,fake_acc_sum/n_bs,loss_D_sum/n_bs,loss_S_D_sum/n_bs,loss_adv_sum/n_bs,loss_pert_sum/n_bs)
            with open(log_file,'a') as log:
                log.write('\n'+out)
            print(out)
            out = "\t\tLoss_G_total:{}, Loss_S_G:{}, Loss_perturb:{}, Loss_adver:{}".format(
                loss_G_sum/n_bs,loss_S_G_sum/n_bs,loss_perturb_sum/n_bs,loss_adver_sum/n_bs)
            with open(log_file,'a') as log:
                log.write('\n'+out)
            print(out)
        ## save final model
        torch.save(self.netG.state_dict(),'netG.pth')
        torch.save(self.netD.state_dict(),'netD.pth')
        torch.save(self.optimizer_G.state_dict(),'optG.pth')
        torch.save(self.optimizer_D.state_dict(),'optD.pth')



    def train_batch(self,images,labels):

        ### D step
        self.optimizer_D.zero_grad()
        self.netD.train()
        self.netG.eval()

        perturb = torch.clamp(self.netG(images),-self.th_perturb,self.th_perturb)
        adv_images = perturb + images
        adv_images = torch.clamp(adv_images,self.box_min,self.box_max)
        d_fake_probs,d_adv_probs = self.netD(adv_images.detach())

        if self._use_attacker:
            pgd_images = self.attacker.perturb(images,labels) 
            d_real_probs,d_pert_probs = self.netD(pgd_images)
        else:
            d_real_probs,d_pert_probs = self.netD(images)

        ## generate labels for discriminator (optionally smooth labels for stability)
        smooth = 0.0
        d_labels_real = torch.ones_like(d_real_probs, device=self.device) * (1 - smooth)
        d_labels_fake = torch.zeros_like(d_fake_probs, device=self.device)
    
        ## discriminator loss
        d_probs = torch.cat((d_real_probs,d_fake_probs),0)
        d_labels = torch.cat((d_labels_real,d_labels_fake),0)
        loss_S_D = F.binary_cross_entropy(d_probs,d_labels)

        loss_adv = F.cross_entropy(d_adv_probs,labels)
        loss_pert = F.cross_entropy(d_pert_probs,labels)

        loss_D = loss_S_D + loss_adv + loss_pert
        loss_D.backward()
        self.optimizer_D.step()


        ### G step
        self.optimizer_G.zero_grad()
        self.netG.train()
        self.netD.eval()

        perturb = torch.clamp(self.netG(images),-self.th_perturb,self.th_perturb)
        adv_images = perturb + images
        adv_images = torch.clamp(adv_images,self.box_min,self.box_max)
        d_fake_probs,d_adv_probs = self.netD(adv_images.detach())

        if self._use_attacker:
            pgd_images = self.attacker.perturb(images,labels) 
            d_real_probs,d_pert_probs = self.netD(pgd_images)
        else:
            d_real_probs,d_pert_probs = self.netD(images)

        ## generate labels for discriminator (optionally smooth labels for stability)
        smooth = 0.0
        d_labels_real = torch.ones_like(d_real_probs, device=self.device) * (1 - smooth)
        d_labels_fake = torch.zeros_like(d_fake_probs, device=self.device)
    
        ## discriminator loss
        d_probs = torch.cat((d_real_probs,d_fake_probs),0)
        d_labels = torch.cat((d_labels_real,d_labels_fake),0)
        loss_S_G = F.binary_cross_entropy(d_probs,d_labels)

        ## calculate perturbation norm
        loss_perturb = torch.norm(perturb.view(perturb.shape[0], -1),2,dim=1)
        loss_perturb = torch.max(loss_perturb - self.th_perturb,torch.zeros(1,device=self.device))
        loss_perturb = torch.mean(loss_perturb)
        
        ## Adv loss
        f_target_logits = self.target_model(adv_images) 
        f_target_probs = F.softmax(f_target_logits,dim=1)
        # if training is targeted, indicate how many examples classified as targets
        # else show accuraccy on adversarial images
        fake_accuracy = torch.mean((torch.argmax(f_target_probs,1)==labels).float())
        onehot_labels = torch.eye(self.target_out_labels, device=self.device)[labels.long()]
        loss_target_adver = adv_loss(f_target_probs,onehot_labels)

        f_dis_probs = F.softmax(d_adv_probs,dim=1)
        onehot_labels = torch.eye(self.target_out_labels, device=self.device)[labels.long()]
        loss_dis_adver = adv_loss(f_dis_probs,onehot_labels)

        loss_adver = loss_target_adver + loss_dis_adver

        alambda = 10.0
        alpha = 1.
        beta = 1.0
        loss_G = alambda*loss_adver - alpha*loss_S_G + beta*loss_perturb
        loss_G.backward()
        self.optimizer_G.step()

        return fake_accuracy.item(),loss_D.item(),loss_S_D.item(),loss_adv.item(),loss_pert.item(),loss_G.item(),loss_S_G.item(), \
                loss_perturb.item(),loss_adver.item()







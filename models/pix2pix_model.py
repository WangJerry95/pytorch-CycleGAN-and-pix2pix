import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(dataset_mode='paired')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--lambda_gp', type=float, default=10.0, help='weight for gp loss')
            parser.add_argument('--unconditioned', action='store_true', default=False,
                                help='if the pix2pix discriminator is conditioned on the input image or not')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'GP']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'interpolated_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if not opt.unconditioned:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                # unconditional discriminator
                self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                              self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.interpolated_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        if not self.opt.unconditioned:
            #conditioned fake
            # stop backprop to the generator by detaching fake_B
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
        else:
            #unconditioned fake
            fake_B = self.fake_AB_pool.query(self.fake_B)
            pred_fake = self.netD(fake_B.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            #unconditioned real
            pred_real = self.netD(self.real_B)
            self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.optimizer_D.zero_grad()
        self.loss_D.backward()

        # interpolate between real_B and fake_B
        alpha = torch.rand(self.real_B.size(0), 1, 1, 1).cuda().expand_as(self.real_B)
        self.interpolated_B = Variable(alpha * self.real_B.data + (1 - alpha) * self.fake_B.data, requires_grad=True)
        # Compute gradient penalty
        if not self.opt.unconditioned:
            interpolated_AB = self.interpolated_B_pool.query(torch.cat((self.real_A, self.interpolated_B), 1))
            pred_interpolated = self.netD(interpolated_AB)
            grad = torch.autograd.grad(outputs=pred_interpolated,
                                       inputs=interpolated_AB,
                                       grad_outputs=torch.ones(pred_interpolated.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
        else:
            interpolated_B = self.interpolated_B_pool.query(self.interpolated_B)
            pred_interpolated = self.netD(interpolated_B)
            grad = torch.autograd.grad(outputs=pred_interpolated,
                                       inputs=interpolated_B,
                                       grad_outputs=torch.ones(pred_interpolated.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

        self.loss_GP = ((grad.norm(2, dim=1)-1) ** 2).mean() * self.opt.lambda_gp
        self.loss_GP.backward(retain_graph=True)


    def backward_G(self):
        if not self.opt.unconditioned:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            # We hope G can fool D , i.e. minimize GAN loss when label is True
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            #unconditioned
            pred_fake = self.netD(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd

import os
import numpy as np

from config import args
from STTransformer import STTransformer_G, STTransformer_D


class cGAN:
    def __init__(
            self,
            adj,
            lambda_term=10.,
            rc_term=10.,
    ):
        # check cuda
        self.device = self.check_cuda()

        # Generator
        self.G = STTransformer_G(
            adj=adj,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.heads,
            time_num=args.time_num
        )
        if args.parallel:
            self.G = nn.DataParallel(self.G, device_ids=range(torch.cuda.device_count()))
        # self.G = torch.load(args.save_path + 'generator.pkl')
        self.G.to(device=self.device)

        # Discriminator
        self.D = STTransformer_D(
            adj=adj,
            nodes=args.nodes,
            features=args.features,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.heads,
            time_num=args.time_num
        )
        if args.parallel:
            self.D = nn.DataParallel(self.D, device_ids=range(torch.cuda.device_count()))
        # self.D = torch.load(args.save_path + 'discriminator.pkl')
        self.D.to(device=self.device)

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=args.lr)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=args.lr)

        # training process
        self.lambda_term = lambda_term
        self.rc_term = rc_term

        # reconstruction loss
        self.MSELoss = nn.MSELoss()

    def train(self, train_x, r):
        #
        for epoch in range(args.epochs):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True
            for q in self.G.parameters():
                q.requires_grad = False

            '''
                update discriminator
            '''

            d_loss = 0.
            for d_iter in range(args.d_iters):
                self.D.zero_grad()

                # sampling with batch
                perm = np.random.permutation(range(r))[: args.batch_size]
                x = train_x[perm]
                x = x.to(device=self.device)

                # noise term
                z = torch.randn(size=[args.batch_size, 1, args.features])
                z = z.expand(args.batch_size, args.nodes, args.features)
                z = z.to(device=self.device)

                # generate samples
                x_tilde = self.G(x + z)

                # backward loss
                # loss of real samples
                d_loss_real = self.D(x)
                d_loss_real = d_loss_real.mean()

                # loss of fake samples
                d_loss_fake = self.D(x_tilde)
                d_loss_fake = d_loss_fake.mean()

                # gradient penalty
                # with torch.backends.cudnn.flags(enabled=False):
                gp = self.calculate_gradient_penalty(x, x_tilde)
                gp = gp.mean()

                d_loss = d_loss_fake - d_loss_real + gp
                d_loss.backward()

                self.d_optimizer.step()

            '''
                update generator
            '''

            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation
            for q in self.G.parameters():
                q.requires_grad = True

            g_loss = 0.
            rc_loss = 0.
            for g_iter in range(args.g_iters):
                self.G.zero_grad()

                # train generator
                # sampling with batch
                perm = np.random.permutation(range(r))[: args.batch_size]
                x = train_x[perm]
                x = x.to(device=self.device)

                # noise term
                z = torch.randn(size=[args.batch_size, 1, args.features])
                z = z.expand(args.batch_size, args.nodes, args.features)
                z = z.to(device=self.device)

                x_tilde = self.G(x + z)  # the generated samples

                # backward loss
                g_loss = self.D(x_tilde)
                g_loss = g_loss.mean()

                # # MSE loss
                mse_loss = self.MSELoss(x, x_tilde)
                rc_loss = mse_loss.mean()

                grc_loss = self.rc_term * rc_loss - g_loss
                grc_loss.backward()

                self.g_optimizer.step()

            if epoch % 1 == 0:
                print('epoch %d: d_loss = %f, g_loss = %f, rc_loss = %f' % (epoch + 1, d_loss, g_loss, rc_loss))
                # Save the trained models
                self.save_model()

    def calculate_gradient_penalty(self, x, x_tilde):
        # linear interpolation
        eps = torch.rand(size=[args.nodes, args.features])
        eps = eps.expand([args.batch_size, args.nodes, args.features])
        eps = eps.to(device=self.device)
        interpolated = eps * x + ((1 - eps) * x_tilde)

        # calculate probability of interpolated examples
        interpolated = Variable(interpolated, requires_grad=True)
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        ones = torch.ones(size=prob_interpolated.size()).to(device=self.device)
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=ones, create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2) * self.lambda_term
        return grad_penalty

    def check_cuda(self):
        # check cuda
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print('use_gpu ==', use_gpu)
            print('device_ids ==', np.arange(0, torch.cuda.device_count()))
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def save_model(self):
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        torch.save(self.G, args.save_path + 'generator.pkl')
        torch.save(self.D, args.save_path + 'discriminator.pkl')

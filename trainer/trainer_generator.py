import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import random
import heapq
from utils.Single_objective_sort import single_sort
from utils.nsga import NSGA_2
from utils.sort import CARS_NSGA
from utils.utils import count_parameters_in_MB
from archs.fully_super_network import Generator
from pytorch_gan_metrics import get_inception_score_and_fid, get_fid, get_inception_score
from torchvision.utils import make_grid, save_image

logger = logging.getLogger(__name__)

try_balance = False
Extra_subnet = 2
num_normal = 7


class GenTrainer():
    def __init__(self, args, gen_net, dis_net, gen_optimizer, dis_optimizer, train_loader, gan_alg, dis_genotype,
                 gen_fixed_genotype):
        self.args = args
        self.gen_net = gen_net
        self.dis_net = dis_net
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.train_loader = train_loader
        self.gan_alg = gan_alg
        self.dis_genotype = dis_genotype
        self.gen_fixed_genotype = gen_fixed_genotype
        self.historical_population = {}
        self.genotypes = np.stack([gan_alg.search() for i in range(args.num_individual)],
                                  axis=0)
        self.best_is_evo = 0
        self.best_fid_evo = 999

    def train(self, epoch, writer_dict, fid_stat, schedulers=None):
        writer = writer_dict['writer']
        gen_step = 0

        # train mode
        gen_net = self.gen_net.train()
        dis_net = self.dis_net.train()

        # number of g
        num_g = int(len(self.gan_alg.Normal_G_fixed[0]) // self.args.num_op_g)
        genotype_G = np.stack([self.gan_alg.search() for i in range(num_g)],
                              axis=0)
        for iter_idx, (imgs, _) in enumerate(tqdm(self.train_loader)):
            global_steps = writer_dict['train_global_steps']

            real_imgs = imgs.type(torch.cuda.FloatTensor)
            i = np.random.randint(0, self.args.num_individual, 1)[0]
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (imgs.shape[0], self.args.latent_dim)))

            # ============================train D-D================================
            self.dis_optimizer.zero_grad()
            # real_validity = dis_net(real_imgs, genotype_D)
            real_validity = dis_net(real_imgs)
            n_i = iter_idx % num_g
            fake_imgs = gen_net(z, genotype_G[n_i]).detach()

            assert fake_imgs.size() == real_imgs.size()
            # fake_validity = dis_net(fake_imgs, genotype_D)
            fake_validity = dis_net(fake_imgs)
            # Hinge loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            self.dis_optimizer.step()
            writer.add_scalar('d_loss', d_loss.item(), global_steps)

            # ===============================train G-G===============================
            if global_steps % self.args.n_critic == 0:
                self.gen_optimizer.zero_grad()
                gen_z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.gen_bs, self.args.latent_dim)))
                for k in range(num_g):
                    gen_imgs = gen_net(gen_z, genotype_G[k])
                    # gen_imgs = gen_net(gen_z, self.gen_fixed_genotype)
                    # fake_validity = dis_net(gen_imgs, genotype_D)
                    fake_validity = dis_net(gen_imgs)
                    # Hinge loss
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                self.gen_optimizer.step()
                genotype_G = np.stack([self.gan_alg.search() for i in range(num_g)],
                                      axis=0)
                # learning rate
                if schedulers:
                    gen_scheduler, dis_scheduler = schedulers
                    g_lr = gen_scheduler.step(global_steps)
                    d_lr = dis_scheduler.step(global_steps)
                    writer.add_scalar('LR/g_lr', g_lr, global_steps)
                    writer.add_scalar('LR/d_lr', d_lr, global_steps)
                writer.add_scalar('g_loss', g_loss.item(), global_steps)
                gen_step += 1
            # verbose
            if gen_step and iter_idx % self.args.print_freq == 0:
                tqdm.write(
                    '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' %
                    (epoch, self.args.max_epoch_D, iter_idx % len(self.train_loader), len(self.train_loader),
                     d_loss.item(), g_loss.item()))
            writer_dict['train_global_steps'] = global_steps + 1
        if epoch % 5 == 0:
            is_warm, is_std_warm, fid_warm = self.validate(self.gen_fixed_genotype, fid_stat)

            logger.info(
                f'IS_warm: {round(is_warm, 3)}, FID_warm: {round(fid_warm, 3)},@ epoch {epoch}.')

    def judege_model_size(self, genotype_G, limit):
        '''
        Test whether the model size constraint is met
        '''
        param_size = count_parameters_in_MB(Generator(self.args, genotype_G))
        if param_size <= limit:
            return True
        return False

    def search(self, parents, fid_stat, num, age_threshold=4):
        '''
        input: parent
        num:
        return:
        '''
        offsprings = self.gen_offspring(parents)
        genotypes = np.concatenate((parents, offsprings), axis=0)
        is_values, fid_values, params = np.zeros(len(genotypes)), np.zeros(
            len(genotypes)), np.zeros(len(genotypes))
        ages = np.zeros(len(genotypes))    #Age
        keep_N, selected_N = len(offsprings), self.args.num_selected
        for idx, genotype_G in enumerate(tqdm(genotypes)):
            encode_G = self.gan_alg.encode(genotype_G)
            if encode_G in self.historical_population:
                data = self.historical_population[encode_G]
                is_value, fid_value, param_size, age = data[0], data[1], data[2], data[3]
                age += 1
                data = self.historical_population[encode_G]
                is_value, fid_value, param_size, age = data
                age += 1  
                self.historical_population[encode_G] = [is_value, fid_value, param_size, age]  # Update the dictionary
            else:
                is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)  # Obtain subnetwork performance
                param_size = count_parameters_in_MB(Generator(self.args, genotype_G))
                age = 1
                self.historical_population[encode_G] = [is_value, fid_value, param_size, age]
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_size
            ages[idx] = age
            if is_value > self.best_is_evo:
                self.best_is_evo = is_value
                file_path = os.path.join(self.args.path_helper['ckpt_path'],
                                         "best_is_gen_.npy")
                np.save(file_path, genotype_G)
            if fid_value < self.best_fid_evo:
                self.best_fid_evo = fid_value
                file_path = os.path.join(self.args.path_helper['ckpt_path'],
                                         "best_fid_gen.npy")
                np.save(file_path, genotype_G)
                
        for i in range(len(genotypes)): #aging mechanism 
            if ages[i] > age_threshold and is_values[i] != self.best_is_evo and fid_values[i] != self.best_fid_evo:
                is_values[i] = 0
                fid_values[i] = 999
            
        sum_is, sum_fid, sum_param = 0, 0, 0
        sum_is, sum_fid, sum_param = np.sum(is_values[:keep_N]), np.sum(fid_values[:keep_N]), np.sum(params[:keep_N])
        avg_is, avg_fid, avg_params = sum_is / keep_N, sum_fid / keep_N, sum_param / keep_N
        logger.info(
            f'mean_IS: {round(avg_is, 3)}, mean_FID: {round(avg_fid, 3)}, max_IS: {round(np.max(is_values), 3)}, min_FID: {round(np.min(fid_values), 3)},mean_params: {round(avg_params, 3)}@ evolutionary_algebra: {num}.')

        keep = NSGA_2(is_values.tolist(), (-fid_values).tolist(), keep_N)
        selected = NSGA_2(is_values.tolist(), (-fid_values).tolist(), selected_N)

        for i in selected:
            logger.info(
                f'genotypes_{i}, IS: {round(is_values[i], 3)}, FID: {round(fid_values[i], 3)}, param_size: {round(params[i], 3)}, age:{round(ages[i], 0)}.')

        self.genotypes = genotypes[keep]
        return genotypes[keep], genotypes[selected], np.mean(is_values[keep]), np.mean(fid_values[keep]), is_values[
            keep], fid_values[keep]

    def validate(self, genotype_G, fid_stat):
        '''
        Calculate IS and FID
        '''
        # eval mode
        gen_net = self.gen_net.eval()
        eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
        fakes = []
        with torch.no_grad():
            for iter_idx in tqdm(range(eval_iter), desc='sample images'):
                z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
                gen_imgs = (gen_net(z, genotype_G) + 1) / 2
                fakes.append(gen_imgs.cpu())
        fakes = torch.cat(fakes, dim=0)

        # get inception score
        (mean, std), fid_score = get_inception_score_and_fid(fakes, fid_stat, verbose=True)
        return mean, std, fid_score

    def gen_offspring(self, alphas, offspring_ratio=1.0):
        """Generate offsprings.
        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """

        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            a, b = np.random.randint(
                0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
            while (a == b):
                a, b = np.random.randint(
                    0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
            alphas_c = self.uniform_crossover_mutation(alphas[a], alphas[b])
            encode_G = self.gan_alg.encode(alphas_c)
            if self.judege_model_size(alphas_c, limit=self.args.max_model_size) and (
                    encode_G not in self.historical_population):
                offsprings.append(alphas_c)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    
    def uniform_crossover_mutation(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        new_alphas = alphas_a.copy()
        for i in range(3):
            for j in range(2):
                p = random.random()
                if p >= 0.5:
                    new_alphas[i][j] = alphas_b[i][j]
            for k in range(5):
                p = random.random()
                if p >= 0.5:
                    new_alphas[i][k + 2] = alphas_b[i][k + 2]
        # Mutation operation
        num_mute = random.randint(1, self.args.mute_max_num)
        for i in range(num_mute):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            if index < 2:
                new_alphas[layer][index] = random.choice(self.gan_alg.Up_G_fixed[2 * layer + index])[0]
            else:
                new_alphas[layer][index] = random.choice(self.gan_alg.Normal_G_fixed[5 * layer + index - 2])[0]
            # print(layer, index)
        return new_alphas

    def get_fitness(self, genotypes, fid_stat):
        '''
        get fid
        '''
        fids = np.zeros(len(genotypes))
        for idx, genotype_G in enumerate(genotypes):
            # eval mode
            gen_net = self.gen_net.eval()
            eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
            fakes = []
            with torch.no_grad():
                for iter_idx in tqdm(range(eval_iter), desc='sample images'):
                    z = torch.cuda.FloatTensor(np.random.normal(
                        0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
                    gen_imgs = (gen_net(z, genotype_G) + 1) / 2
                    fakes.append(gen_imgs.cpu())
            fakes = torch.cat(fakes, dim=0)
            fid_score = get_fid(fakes, fid_stat)
            # fid = get_fid()
            fids[idx] = fid_score
        return fids

    def get_index(self, genotypes, fid_stat):
        # Obtain a ranking of performance from good to bad
        fids_G = self.get_fitness(genotypes, fid_stat)
        index = fids_G.argsort()
        return index

    def get_operation_score(self, genotypes, index):
        '''
        Obtain scores for each operation
        genotypes:Genes obtained from sampling
        index:Sorted Index
        '''
        scores_up = np.zeros((3, 2, 3), dtype=int)
        scores_normal = np.zeros((3, 5, 7), dtype=int)
        jj = 0
        num = len(genotypes) / 2
        for kk in index:
            for i in range(3):
                for j in range(2):
                    scores_up[i][j][genotypes[kk][i][j]] += 1
                for k in range(2):
                    scores_normal[i][k][genotypes[kk][i][k + 2]] += 1
                for k in range(2, 5):
                    scores_normal[i][k][genotypes[kk][i][k + 2]] += 1
            jj += 1
            if jj == num:
                break
        return scores_up, scores_normal

    def calculate_operation_weights(self, scores):
        """
        Calculate the normalized weights for operations based on their scores.
        
        Args:
        scores (numpy array): A 3D numpy array of scores, where each element represents 
                            the count of an operation's occurrence.

        Returns:
        numpy array: A 3D numpy array of weights, corresponding to the scores array.
        """
        weights = np.zeros_like(scores, dtype=float)
        
        for i in range(scores.shape[0]): 
            for j in range(scores.shape[1]):  
                total = np.sum(scores[i][j])
                if total > 0:
                    weights[i][j] = scores[i][j] / total
                else:
                    weights[i][j] = 1.0 / scores.shape[2]

        return weights


    def directly_modify_fixed(self, fid_stat):
        #  clear
        self.clear_bag()
        # sample
        genotypes_to_eva = np.stack([self.gan_alg.search() for i in range(42)], axis=0)
        rank = self.get_index(genotypes_to_eva, fid_stat)  # get rank
        scores_up, scores_normal = self.get_operation_score(genotypes_to_eva, rank)
        weights_up = self.calculate_operation_weights(scores_up)
        weights_normal = self.calculate_operation_weights(scores_normal)
        for i in range(len(self.gan_alg.Normal_G_fixed)):
            self.gan_alg.Normal_G_fixed[i] = [(op, weights_normal[i // self.gan_alg.normal_nodes][i % self.gan_alg.normal_nodes][op]) 
                                    for op in range(len(self.gan_alg.Normal_G_fixed[i]))]

        for i in range(len(self.gan_alg.Up_G_fixed)):
            self.gan_alg.Up_G_fixed[i] = [(op, weights_up[i // self.gan_alg.up_nodes][i % self.gan_alg.up_nodes][op]) 
                                for op in range(len(self.gan_alg.Up_G_fixed[i]))]
        self.clear_bag()
        self.genotype_fixG = self.gan_alg.search(remove=False)

    def clear_bag(self):
        self.gan_alg.Normal_G = []
        for i in range(15):
            self.gan_alg.Normal_G.append([])
        self.gan_alg.Up_G = []
        for i in range(6):
            self.gan_alg.Up_G.append([])


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

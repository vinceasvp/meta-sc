import os
import time
import numpy as np
from utils.utils import ensure_path
from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
import torch
from .helper import *
from dataloader.dataloader import get_base_dataloader_meta, get_new_dataloader, get_pretrain_dataloader, get_task_specific_testloader, get_testloader, get_novel_testloader
import torch.distributions.normal as normal

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.set_up_datasets()

        self.model = MYNET(self.args, mode=self.args.network.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.old_model = MYNET(self.args, mode=self.args.network.base_mode)
        self.old_model = nn.DataParallel(self.old_model, list(range(self.args.num_gpu)))
        self.old_model = self.old_model.cuda()

        if self.args.model_dir.s3c_model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir.s3c_model_dir)
            self.best_model_dict = torch.load(self.args.model_dir.s3c_model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())
        
        self.args.episode.episode_way = 5
        self.args.episode.episode_shot = 5

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.optimizer.decay)
        
        if self.args.scheduler.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler.step, gamma=self.args.scheduler.gamma)
        elif self.args.scheduler.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.scheduler.milestones,
                                                             gamma=self.args.scheduler.gamma)
        elif self.args.scheduler.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs.epochs_base)
        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, valset, trainloader, valloader = get_base_dataloader_meta(self.args)
            # trainset, valset, trainloader, valloader = get_pretrain_dataloader(self.args)
        else:
            trainset, valset, trainloader, valloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, valloader
    """
    def get_novel_dataloader(self, session):
        if session == 0:
            _, _, testloader = get_base_dataloader(self.args)
        else:
            testloader = get_novel_test_dataloader(self.args, session)
        return testloader

    def get_task_specific_test_dataloader(self, session):
        if session == 0:
            _, _, testloader = get_base_dataloader(self.args)
        else:
            testloader = get_task_specific_new_dataloader(self.args, session)
        return testloader
    """

    def train(self):
        args = self.args
        t_start_time = time.time()
        # init train statistics
        self.result_list = [args]
        if args.start_session == 0:
            session = 0
            train_set, trainloader, testloader = self.get_dataloader(session=0)
            self.model.load_state_dict(self.best_model_dict)
            best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            print('new classes for this session:\n', np.unique(train_set.targets))
            optimizer, scheduler = self.get_optimizer_base()
            #self.best_model_dict = torch.load(best_model_dir)['params']
            #self.model.load_state_dict(self.best_model_dict)
            for epoch in range(args.epochs.epochs_base):
                start_time = time.time()
                
            
                # train base sess
                tl, ta, treg = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                # test model with all seen class
                tsl, tsa, _, acc_dict = test_agg(self.model, testloader, epoch, args, session)
                self.sess_acc_dict[f'sess {session}'] = acc_dict

                # save better model
                if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    self.trlog['max_acc_epoch'] = epoch
                    save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    print('********A better model is found!!**********')
                    print('Saving model to :%s' % save_model_dir)
                print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                    self.trlog['max_acc'][session]))

                self.trlog['train_loss'].append(tl)
                self.trlog['train_acc'].append(ta)
                self.trlog['test_loss'].append(tsl)
                self.trlog['test_acc'].append(tsa)
                lrc = scheduler.get_last_lr()[0]
                self.result_list.append(
                    'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                        epoch, lrc, tl, ta, tsl, tsa))
                print('epoch:%03d,lr:%.4f,training_ce_loss:%.5f, training_reg_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                        epoch, lrc, tl, treg,ta, tsl, tsa))
                print('This epoch takes %d seconds' % (time.time() - start_time),
                        '\nstill need around %.2f mins to finish this session' % (
                                (time.time() - start_time) * (args.epochs.epochs_base - epoch) / 60))
                scheduler.step()

            self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
            
            if args.strategy.data_init:

                #print("Went inside #################")
                print("Updating old class with class means ")
                self.model.load_state_dict(self.best_model_dict)
                #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                session0_best_model_dict = deepcopy(self.model.state_dict())
                torch.save(dict(params=self.model.state_dict()), best_model_dir)

                self.model.module.mode = 'avg_cos'
                tsl, tsa, _ , acc_dict = test_agg(self.model, testloader, 0, args, session)

                self.sess_acc_dict[f'sess {session}'] = acc_dict
                if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

        acc_array = np.zeros([self.args.test_times, 4, self.args.num_session], dtype=float)
        for i in range(self.args.test_times):
            tmp_acc_df = self.evaluate(session0_best_model_dict)
            acc_array[i] = tmp_acc_df.to_numpy()
        acc_array = acc_array.mean(0)


        print("\n\n\nFinal result:")
        self.result_list.append("\n\n\nFinal result:")
        cpi, msr_overall, acc_aver_df, ar_over = cal_auxIndex_from_numpy(acc_array)
        pd = acc_array[3][0] - acc_array[3][-1]
        indexes = {'PD':pd, 'CPI':cpi, 'AR':ar_over, 'MSR':msr_overall}
        indexes_df = pandas.DataFrame.from_dict(indexes, orient='index')
        final_df = pandas.DataFrame(acc_array)
        # final_df = final_df.T
        # pretty output
        pandas.set_option('display.max_rows', None)
        pandas.set_option('display.max_columns', None)
        pandas.set_option('display.width', None)
        pandas.set_option('display.max_colwidth', None)

        excel_fn = os.path.join(self.args.save_path, "output.xlsx")
        print("save output at ", excel_fn)
        writer = pandas.ExcelWriter(excel_fn)
        final_df.to_excel(writer, sheet_name='final_df')
        acc_aver_df.to_excel(writer, sheet_name='final_df', startrow=7)
        indexes_df.to_excel(writer, sheet_name='final_df', startrow=13)
        indexes_df.T.to_excel(writer, sheet_name='final_df', startrow=20)
        writer.save()

        output = f"\nreslut on {self.args.dataset}, method {self.args.project}\
                    \n{self.args.save_path}\
                    \n****************************************Pretty Output********************************************\
                    \n{final_df}\
                    \n===> Comprehensive Performance Index(CPI) v2: {cpi}\n===> PD: {pd}\
                    \n===> Memory Strock Ratio(MSR) Overall: {msr_overall}\n===> Amnesia Rate(AR): {ar_over}\
                    \n===> Acc Average: \n{acc_aver_df}\
                    \n***********************************************************************************************"
        print(output)
        self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)
        
    def evaluate(self, session0_best_model_dict):
        args = self.args
        for session in range(args.start_session, args.num_session):

            train_set, trainloader, testloader = self.get_dataloader(session)
            test_set, testloader = get_testloader(self.args, session)


            # self.model.load_state_dict(self.best_model_dict)
            self.model.load_state_dict(session0_best_model_dict)
            best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')

            if session == 0:  # load base class train img label
                print('test classes for this session:\n', np.unique(test_set.targets))

            else:  # incremental learning sessions
                
                print("training session: [%d]" % session)
                previous_class = (args.num_base + (session-1) * args.way) 
                present_class = (args.num_base + session * args.way) 

                self.model.module.mode = self.args.network.new_mode
                self.model.eval()        
                
                self.model.train()
                for parameter in self.model.module.parameters():
                    parameter.requires_grad = False
                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                for parameter in self.model.module.fc.parameters():
                    parameter.requires_grad = True
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)

                optimizer = torch.optim.SGD(self.model.parameters(),lr=self.args.lr.lr_new, momentum=0.9, dampening=0.9 , weight_decay=0)

                support_data, support_label = self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                print('Started fine tuning')
                T = 2
                beta = 0.25
                with torch.enable_grad():
                    for epoch in range(self.args.epochs.epochs_new):
                        inputs, label = support_data, support_label
                    
                        logits, feature, attention_op = self.model(inputs, stochastic = False)
                        
                        protos = args.proto_list
                        indexes = torch.randperm(protos.shape[0])
                        protos = protos[indexes]
                        temp_protos = protos.cuda()
                        
                        num_protos = temp_protos.shape[0] 
                        
                        label_proto = torch.arange(previous_class).cuda()
                        label_proto = label_proto[indexes]
                        
                        temp_protos = torch.cat((temp_protos,feature))
                        label_proto = torch.cat((label_proto,label))
                        logits_protos = self.model.module.fc(temp_protos, stochastic=args.stochastic) # True
                        ############################

                        loss_proto = nn.CrossEntropyLoss()(logits_protos[:num_protos,:present_class], label_proto[:num_protos]) * args.lamda_proto 
                        loss_ce = nn.CrossEntropyLoss()(logits_protos[num_protos:, :present_class], label_proto[num_protos:] ) * (1 - args.lamda_proto)
                        
                        optimizer.zero_grad()
                        
                        loss = loss_proto + loss_ce
                        loss.backward()
                        optimizer.step()

                        print('Epoch: {}, Loss_CE: {}, Loss proto:{}, Loss: {}'.format(epoch, loss_ce, loss_proto, loss))
                        #print(c)
                
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                #########################################################################            
        
            ################  Printing performance metrics ##################
            self.model.module.mode = self.args.network.new_mode
            self.model.eval()
            


            tsl, tsa, tsa_agg, acc_dict = test_agg(self.model, testloader, 0, args, session, print_numbers=True, save_pred=True)
            self.sess_acc_dict[f'sess {session}'] = acc_dict
            print('Overall cumulative accuracy: {}, after agg: {}'.format(tsa*100, tsa_agg*100))
            self.result_list.append('Current Session {}, overall cumulative accuracy: {}, after agg: {}'.format(session, tsa*100, tsa_agg*100))
            # save_features(self.model, testloader, args, session)
            
            testset, novel_testloader = get_novel_testloader(self.args, session)
            tsl_novel, tsa_novel, tsa_agg_novel, acc_dict= test_agg(self.model, novel_testloader, 0, args, session)
            print('Novel classes cumulative accuracy: {}, after agg: {}'.format(tsa_novel*100, tsa_agg_novel*100))
            self.result_list.append('Current Session {}, novel classes cumulative accuracy: {}, after agg: {}'.format(session, tsa_novel*100, tsa_agg_novel*100))
            hm = 0
            hm_agg = 0
            hm_agg_stoc_agg = 0
            for j in range(0,session+1):
                testset, specific_testloader = get_task_specific_testloader(self.args, j)
                tsl, tsa,tsa_agg, acc_dict = test_agg(self.model, specific_testloader, 0, args, session)
                if session ==0:
                    tsa_base = tsa
                    tsa_agg_base = tsa_agg
                print('session: {} test accuracy: {}, after agg: {}'.format(j, tsa * 100, tsa_agg*100))
                self.result_list.append('session: {} test accuracy: {}, after agg: {}'.format(j, tsa * 100, tsa_agg*100))
                hm += 1/((tsa+0.0000000001)*100)
                hm_agg += 1/((tsa_agg+0.00000001)*100)
            if session>0:    
                print('Task wise Harmonic mean is : {}, agg: {}'.format((session+1)/hm, (session+1)/hm_agg))
                self.result_list.append('Task wise Harmonic mean is : {}, agg: {}'.format((session+1)/hm, (session+1)/hm_agg))
                hm = (2*tsa_base*tsa_novel)/(tsa_base+0.00000001+tsa_novel)
                hm_agg = ((2*tsa_agg_base*tsa_agg_novel)/(tsa_agg_base+tsa_agg_novel+0.00000001))
                print('Harmonic mean between old and new classes : {}, agg: {}'.format(hm*100,hm_agg*100))
                self.result_list.append('Harmonic mean between old and new classes : {}, agg: {}'.format(hm*100,hm_agg*100))
            ###################################################################
            
            ############ Update protos and save features #####################
            if session == 0:
                update_sigma_protos_feature_output(trainloader, train_set, self.model, args, session)
            else:
                update_sigma_novel_protos_feature_output(support_data, support_label, self.model, args, session)

            print('protos, radius', args.proto_list.shape, args.radius)
            self.result_list.append('protos {}, radius {}'.format(args.proto_list.shape, args.radius))
        
            self.model.module.mode = self.args.network.new_mode
            output, acc_df, final_df = self.pretty_output(save=False, print_output=False)
            save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)
        self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)
        ##################################################################                
        return final_df

    def set_save_path(self):
        mode = self.args.network.base_mode + '-' + self.args.network.new_mode
        if self.args.strategy.data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        self.args.save_path = self.args.save_path + str(self.args.stochastic) + '_stochastic' \
                                + 'lamda_proto' +str(self.args.lamda_proto) + 'way'+str(self.args.way)+\
                                    'shot' + str(self.args.shot) + '_%s/' % self.args.Method  
        if self.args.scheduler.schedule == 'Milestone':
            mile_stone = str(self.args.scheduler.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs.epochs_base, self.args.lr.lr_base, mile_stone, self.args.scheduler.gamma, self.args.batch_size_base,
                self.args.optimizer.momentum)
        elif self.args.scheduler.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs.epochs_base, self.args.lr.lr_base, self.args.scheduler.step, self.args.scheduler.gamma, self.args.batch_size_base,
                self.args.optimizer.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.network.temperature)

        if 'ft' in self.args.network.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr.lr_new, self.args.epochs.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

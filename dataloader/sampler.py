import torch
import numpy as np
import copy
import math

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

class DFSLTrainCategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_inc_cls, n_shot, n_query):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_inc_cls = n_inc_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.inc_n_per = n_shot + n_query
        self.n_base_test_samples = n_inc_cls * n_query
        self.all_cls = np.arange(n_cls)

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            # ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            fs_batch = []
            base_batch = []
            inc_classes = torch.randperm(len(self.m_ind))[:self.n_inc_cls]  # random sample num_class indexs,e.g. 5

            for c in inc_classes:
                l = torch.from_numpy(self.m_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.inc_n_per]  # sample n_per data index of this class
                fs_batch.append(l[pos])
            fs_batch = torch.stack(fs_batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd

            left_classes = np.delete(self.all_cls, inc_classes.numpy())
            KleftIndices = np.random.choice(
                left_classes, size=self.n_base_test_samples, replace=True)
            unique_classes, NumImagesPerCategory = np.unique(
                KleftIndices, return_counts=True)

            for i, c in enumerate(torch.from_numpy(unique_classes)):
                l = torch.from_numpy(self.m_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:NumImagesPerCategory[i]]  # sample n_per data index of this class
                base_batch.append(l[pos])
            base_batch = torch.concat(base_batch)
            batch = torch.concat([fs_batch, base_batch])
            yield batch



class MineTrainCategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_inc_cls, n_shot, n_query):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_inc_cls = n_inc_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.inc_n_per = n_shot + n_query
        self.n_base_test_samples = n_inc_cls * n_query
        self.all_cls = np.arange(n_cls)

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            # ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            fs_batch = []
            base_batch = []
            inc_classes = torch.randperm(len(self.m_ind))[:self.n_inc_cls]  # random sample num_class indexs,e.g. 5

            for c in inc_classes:
                l = torch.from_numpy(self.m_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.inc_n_per]  # sample n_per data index of this class
                fs_batch.append(l[pos])
            fs_batch = torch.stack(fs_batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd

            left_classes = np.delete(self.all_cls, inc_classes.numpy())

            for i, c in enumerate(torch.from_numpy(left_classes)):
                l = torch.from_numpy(self.m_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_query]  # sample n_per data index of this class
                base_batch.append(l[pos])
            base_batch = torch.concat(base_batch)
            batch = torch.concat([fs_batch, base_batch])
            yield batch
            """
            for i in range(math.ceil(self.n_cls / self.n_inc_cls)):
                if (i+1) * self.n_inc_cls * self.inc_n_per <= len(batch):
                    yield batch[i * self.n_inc_cls * self.inc_n_per: (i+1) * self.n_inc_cls * self.inc_n_per]
                else:
                    yield batch[i * self.n_inc_cls * self.inc_n_per:]
            """


class TrueIncreTrainCategoriesSampler():

    def __init__(self, label, n_batch, na_base_cls, na_inc_cls, np_base_cls, np_inc_cls, nb_shot, nn_shot, n_query):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.na_base_cls = na_base_cls
        self.na_inc_cls = na_inc_cls
        self.np_base_cls = np_base_cls
        self.np_inc_cls = np_inc_cls
        self.nb_shot = nb_shot
        self.nn_shot = nn_shot
        self.n_query = n_query
        self.base_samples_per_cls = nb_shot + n_query
        self.novel_samples_per_cls = nn_shot + n_query
        # self.n_base_test_samples = np_inc_cls * n_query
        self.all_cls = np.arange(na_base_cls + na_inc_cls)

        label = np.array(label)  # all data label
        self.tmp_base_ind = []  # the data index of each temp base class
        for i in range(self.na_base_cls):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            # ind = torch.from_numpy(ind)
            self.tmp_base_ind.append(ind)

        self.tmp_incre_ind = []  # the data index of each incremental train class
        for i in range(self.na_base_cls, self.na_base_cls + self.na_inc_cls):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            # ind = torch.from_numpy(ind)
            self.tmp_incre_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            base_batch = []
            tmp_base_classes = torch.randperm(len(self.tmp_base_ind))[:self.np_base_cls]
            for c in tmp_base_classes:
                l = torch.from_numpy(self.tmp_base_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.base_samples_per_cls]  # sample n_per data index of this class
                base_batch.append(l[pos])
            base_batch = torch.stack(base_batch).t().reshape(-1)

            incre_fs_batch = []
            inc_classes = torch.randperm(len(self.tmp_incre_ind))[:self.np_inc_cls]  # random sample num_class indexs,e.g. 5
            for c in inc_classes:
                l = torch.from_numpy(self.tmp_incre_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.novel_samples_per_cls]  # sample n_per data index of this class
                incre_fs_batch.append(l[pos])
            incre_fs_batch = torch.stack(incre_fs_batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd

            batch = torch.concat([base_batch, incre_fs_batch])
            yield batch



class SupportsetSampler():

    def __init__(self, label, n_cls, n_per,  n_batch=1, seq_sample=False):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        self.seq_sample = seq_sample
        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(min(label), max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            assert len(self.m_ind) == self.n_cls
            if self.seq_sample:
                classes =  list(range(len(self.m_ind)))[:self.n_cls]
            else:
                classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                if self.seq_sample:
                    pos = list(range(len(l)))[:self.n_per]
                else:
                    pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
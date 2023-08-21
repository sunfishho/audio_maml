from    birdCall import BirdCall
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np
import random
import pdb

class BirdCallNShot:

    NUM_TAKEN_FROM_SPECIES = 100
    TRAIN_RATIO = 0.75

    def __init__(self, root, batchsz, n_way, k_shot, k_query):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        """
        
        self.dataset = BirdCall()

        self.batchsz = batchsz
        self.n_cls = len(self.dataset.most_represented_birds)  # varies based on self.dataset.MIN_RECORDINGS
        
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query

        self.x_preselected = np.load(self.dataset.data_output_path)
        self.y_preselected = np.load(self.dataset.labels_output_path)

        iteration_idx = 0

        self.num_species_train = round(self.n_cls * self.TRAIN_RATIO)
        self.num_species_test = self.n_cls - self.num_species_train

        assert (k_shot + k_query) <= self.NUM_TAKEN_FROM_SPECIES
        # assert self.num_species_train >= self.n_way and self.num_species_test >= self.n_way, "Not enough species for n-way classification!"

        # The 1 is a hack, we will later permute these such that that dimension is the last dimension
        self.x = np.zeros([self.n_cls, 1, self.NUM_TAKEN_FROM_SPECIES, 48, 128])
        self.x_train = np.zeros([self.num_species_train, 1, self.NUM_TAKEN_FROM_SPECIES, 48, 128])
        self.x_test = np.zeros([self.num_species_test, 1, self.NUM_TAKEN_FROM_SPECIES, 48, 128])

        # Pick self.NUM_TAKEN_FROM_SPECIES recordings from each bird species to create a balanced dataset

        num_species_filled = 0
        
        while iteration_idx < self.x_preselected.shape[0]:
            bird_idx = self.y_preselected[iteration_idx]
            bird_name = self.dataset.idx_to_name[bird_idx]
            selections = random.sample(range(self.dataset.bird_count[bird_name]), self.NUM_TAKEN_FROM_SPECIES)
            for recording_idx, offset in enumerate(selections):
                self.x[num_species_filled, 0, recording_idx] = np.copy(self.x_preselected[iteration_idx + offset])
                assert bird_idx == self.y_preselected[iteration_idx + offset]
                # Put first self.num_species_train species into train, and the rest into test
                if num_species_filled < self.num_species_train:
                    self.x_train[num_species_filled, 0, recording_idx] = np.copy(self.x_preselected[iteration_idx + offset])
                else:
                    self.x_test[num_species_filled - self.num_species_train, 0, recording_idx] = np.copy(self.x_preselected[iteration_idx + offset])
                    
            iteration_idx += self.dataset.bird_count[bird_name]
            num_species_filled += 1
    
        self.x = np.transpose(self.x, [0,2,3,4,1])
        self.x_train = np.transpose(self.x_train, [0,2,3,4,1])
        self.x_test = np.transpose(self.x_test, [0,2,3,4,1])

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

        
    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(self.NUM_TAKEN_FROM_SPECIES, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, 48, 128)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, 48, 128)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, 48, 128)
            y_spts = np.array(y_spts).astype(np.int32).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, 48, 128)
            y_qrys = np.array(y_qrys).astype(np.int32).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch





if __name__ == '__main__':
    
    print("This should not be called")

    # import  time
    # import  torch
    # import  visdom

    # # plt.ion()
    # viz = visdom.Visdom(env='omniglot_view')

    # db = OmniglotNShot('db/omniglot', batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=64)

    # for i in range(1000):
    #     x_spt, y_spt, x_qry, y_qry = db.next('train')


    #     # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
    #     x_spt = torch.from_numpy(x_spt)
    #     x_qry = torch.from_numpy(x_qry)
    #     y_spt = torch.from_numpy(y_spt)
    #     y_qry = torch.from_numpy(y_qry)
    #     batchsz, setsz, c, h, w = x_spt.size()


    #     viz.images(x_spt[0], nrow=5, win='x_spt', opts=dict(title='x_spt'))
    #     viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
    #     viz.text(str(y_spt[0]), win='y_spt', opts=dict(title='y_spt'))
    #     viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


    #     time.sleep(10)


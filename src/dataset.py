import numpy as np

class Dataset_4:
    def __init__(self,data1,data2,data3,data4):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data1 = data1
        self._data2 = data2
        self._data3 = data3
        self._data4 = data4
        self._num_examples = data1.shape[0]
        pass

    @property
    def data1(self):
        return self._data1
    @property
    def data2(self):
        return self._data2
    @property
    def data3(self):
        return self._data3
    @property
    def data4(self):
        return self._data4

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data1 = self.data1[idx]  # get list of `num` random samples
            self._data2 = self.data2[idx]
            self._data3 = self.data3[idx]
            self._data4 = self.data4[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data1_rest_part = self.data1[start:self._num_examples]
            data2_rest_part = self.data2[start:self._num_examples]
            data3_rest_part = self.data3[start:self._num_examples]
            data4_rest_part = self.data4[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data1 = self.data1[idx0]  # get list of `num` random samples
            self._data2 = self.data2[idx0]
            self._data3 = self.data3[idx0]
            self._data4 = self.data4[idx0]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch
            data1_new_part = self._data1[start:end]
            data2_new_part = self._data2[start:end]
            data3_new_part = self._data3[start:end]
            data4_new_part = self._data4[start:end]
            return_data1 = np.concatenate((data1_rest_part, data1_new_part), axis=0)
            return_data2 = np.concatenate((data2_rest_part, data2_new_part), axis=0)
            return_data3 = np.concatenate((data3_rest_part, data3_new_part), axis=0)
            return_data4 = np.concatenate((data4_rest_part, data4_new_part), axis=0)
            return return_data1, return_data2, return_data3, return_data4
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data1[start:end], self._data2[start:end], self._data3[start:end], self._data4[start:end]

class Dataset_3:
    def __init__(self,data1,data2,data3):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data1 = data1
        self._data2 = data2
        self._data3 = data3
        self._num_examples = data1.shape[0]
        pass

    @property
    def data1(self):
        return self._data1
    @property
    def data2(self):
        return self._data2
    @property
    def data3(self):
        return self._data3

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data1 = self.data1[idx]  # get list of `num` random samples
            self._data2 = self.data2[idx]
            self._data3 = self.data3[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data1_rest_part = self.data1[start:self._num_examples]
            data2_rest_part = self.data2[start:self._num_examples]
            data3_rest_part = self.data3[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data1 = self.data1[idx0]  # get list of `num` random samples
            self._data2 = self.data2[idx0]
            self._data3 = self.data3[idx0]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch
            data1_new_part = self._data1[start:end]
            data2_new_part = self._data2[start:end]
            data3_new_part = self._data3[start:end]
            return_data1 = np.concatenate((data1_rest_part, data1_new_part), axis=0)
            return_data2 = np.concatenate((data2_rest_part, data2_new_part), axis=0)
            return_data3 = np.concatenate((data3_rest_part, data3_new_part), axis=0)
            return return_data1, return_data2, return_data3
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data1[start:end], self._data2[start:end], self._data3[start:end]

class Dataset_2:
    def __init__(self, data1, data2):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data1 = data1
        self._data2 = data2
        self._num_examples = data1.shape[0]
        pass

    @property
    def data1(self):
        return self._data1

    @property
    def data2(self):
        return self._data2


    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data1 = self.data1[idx]  # get list of `num` random samples
            self._data2 = self.data2[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data1_rest_part = self.data1[start:self._num_examples]
            data2_rest_part = self.data2[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data1 = self.data1[idx0]  # get list of `num` random samples
            self._data2 = self.data2[idx0]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            data1_new_part = self._data1[start:end]
            data2_new_part = self._data2[start:end]
            return_data1 = np.concatenate((data1_rest_part, data1_new_part), axis=0)
            return_data2 = np.concatenate((data2_rest_part, data2_new_part), axis=0)
            return return_data1, return_data2,
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data1[start:end], self._data2[start:end]

#
# dataset= Dataset_2(np.arange(0, 20), np.arange(0, 20))
#
# for i in range(10):
#     dataset1, dataset2 = dataset.next_batch(6)
#     print(dataset1)
#     print(dataset2)
#     print('------------')
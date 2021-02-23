import numpy as np


class Sampler(object):
    def __init__(self):
        pass


class RandomSampler(Sampler):
    def __init__(self, data, output_info):
        super(RandomSampler, self).__init__()
        self.data = data
        self.n = len(data)
        self.model = self.build_sample_model(data, output_info)

    def build_sample_model(self, data, output_info):
        sample_model = []
        st = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    # Skip continuous data one hot vector
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])  # The indice of non-zero elements

                sample_model.append(tmp)
                st = ed
            else:
                assert 0

        assert st == data.shape[1]

        return sample_model

    def sample(self, n, col=None, opt=None):
        """
            n: The size of a batch, e.g. 500
            col: [batch_size, ], e.g. (500,)
                 The index of categorial column is selected among categorial columns
            opt: [batch_size, ], e.g. (500,)
                 The index of category is selected in a categorial column
        """
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]

        idx = []  # batch_size of list
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))

        return self.data[idx]

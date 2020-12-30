import numpy as np


class Sampler(object):
    def __init__(self):
        pass


class RandomSampler(Sampler):
    def __init__(self, data, output_info):
        super(RandomSampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)

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

                self.model.append(tmp)
                st = ed
            else:
                assert 0

        assert st == data.shape[1]

    def sample(self, n, col, opt):
        """
            n:
            col:
            opt:
        """
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]

        idx = []  # batch_size of list
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))

        return self.data[idx]

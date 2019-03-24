import codecs
import random

def load_tri_sentences(path):
    expand, sens, sen = list(), list(), list()
    c, l, t, a = list(), list(), list(), list()
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if line:
            word = line.split()
            c.append(int(word[0]))
            l.append(int(word[1]))
            t.append(int(word[2]))
            a.append(int(word[3]))
        else:
            if len(c) > 0:
                sens.append([c, l, t, a])
                c, l, t, a = [], [], [], []
    for x in range(len(sens)):
        sen = sens[x]
        for y in range(len(sen[0])):
            if y > 0:
                mask = [1 for i in range(y)]
                mask += [2 for i in range(len(sen[0]) - y)]
                cut = y
                tri_loc = []
                for i in range(len(sen[0])):
                    tri_loc.append(i - cut)
                tri_in = [0 for i in range(34)]
                tri_in[sen[2][y]] = 1
                expand.append([sen[0], sen[1], sen[2], sen[3], tri_in, tri_loc, mask, cut])
    return expand

def load_arg_sentences(path):
    expand, sens, sen = list(), list(), list()
    c, l, t, a = list(), list(), list(), list()
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if line:
            word = line.split()
            c.append(int(word[0]))
            l.append(int(word[1]))
            t.append(int(word[2]))
            a.append(int(word[3]))
        else:
            if len(c) > 0:
                sens.append([c, l, t, a])
                c, l, t, a = [], [], [], []
    for x in range(len(sens)):
        sen = sens[x]
        tri_f = 0
        for i in range(len(sen[0])):
            if sen[2][i] != 0:
                tri_f = i
                break
        for y in range(len(sen[0])):
            if y > 0 and y != tri_f:
                fir = min(tri_f, y)
                sec = max(tri_f, y)
                mask = [1 for i in range(fir)]
                mask += [2 for i in range(sec - fir)]
                mask += [3 for i in range(len(sen[0]) - sec)]
                cut = [tri_f, y]
                tri_loc, arg_loc = [], []
                for i in range(len(sen[0])):
                    tri_loc.append(i - cut[0])
                    arg_loc.append(i - cut[1])
                arg_in = [0 for i in range(36)]
                arg_in[sen[3][y]] = 1
                # if len(sen[0]) > 75:
                #     print(str(len(sen[0])) + '   ' + str(len(mask)))
                expand.append([sen[0], sen[1], sen[2], sen[3], arg_in, tri_loc, arg_loc, mask, cut])
    print("load_finished!")
    return expand


class Batch_tri(object):
    def __init__(self, data, batch_size, sen_len):
        self.batch_data = self.sort_pad(data, batch_size, sen_len)
        self.len_data = len(self.batch_data)
        self.length = int(sen_len)

    def sort_pad(self, data, batch_size, sen_len):
        num_batch = int(len(data)/batch_size)
        sort_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad(sort_data[i * batch_size: (i + 1) * batch_size], sen_len))
        return batch_data

    @staticmethod
    def pad(data, length):
        chars, ls, tri, arg, tri_in, tri_loc, mask, cut = list(), list(), list(), list(), list(), list(), list(), list()
        for line in data:
            c, l, t, a, t_i, t_l, m, cu = line
            padding = [0] * (length - len(c))
            chars.append(c + padding)
            ls.append(l + padding)
            tri.append(t + padding)
            arg.append(a + padding)
            tri_in.append(t_i)
            mask.append(m + [0] * (length - len(c) - 2))
            cut.append(cu)
            for i in range(length - len(c)):
                t_l.append(t_l[len(c) - 1] + i + 1)
            for i in range(len(c)):
                t_l[i] += length - 1
            tri_loc.append(t_l)
        return [chars, ls, tri, arg, tri_in, tri_loc, mask, cut]

    def iter_batch(self):
        random.shuffle(self.batch_data)
        for i in range(self.len_data):
            yield self.batch_data[i]

class Batch_arg(object):
    def __init__(self, data, batch_size, sen_len):
        self.batch_data = self.sort_pad(data, batch_size, sen_len)
        self.len_data = len(self.batch_data)
        self.length = int(sen_len)

    def sort_pad(self, data, batch_size, sen_len):
        num_batch = int(len(data)/batch_size)
        sort_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad(sort_data[i * batch_size: (i + 1) * batch_size], sen_len))
        return batch_data

    @staticmethod
    def pad(data, length):
        chars, ls, tri, arg, arg_in, tri_loc, arg_loc, mask, cut = list(), list(), list(), list(), list(), list(), list(), list(), list()
        for line in data:
            c, l, t, a, a_i, t_l, a_l, m, cu = line
            padding = [0] * (length - len(c))
            chars.append(c + padding)
            ls.append(l + padding)
            tri.append(t + padding)
            arg.append(a + padding)
            arg_in.append(a_i)
            mask.append(m + [0] * (length - len(c) - 2))
            cut.append(cu)
            for i in range(length - len(c)):
                t_l.append(t_l[len(c) - 1] + i + 1)
                a_l.append(a_l[len(c) - 1] + i + 1)
            for i in range(len(c)):
                t_l[i] += length - 1
                a_l[i] += length - 1
            tri_loc.append(t_l)
            arg_loc.append(a_l)
        return [chars, ls, tri, arg, arg_in, tri_loc, arg_loc, mask, cut]

    def iter_batch(self):
        random.shuffle(self.batch_data)
        for i in range(self.len_data):
            yield self.batch_data[i]
import os
import random
import pickle
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from stanfordcorenlp import StanfordCoreNLP

def read_filename():
    file_name = list()
    train_sgm, train_apf, test_sgm, test_apf, dev_sgm, dev_apf = list(), list(), list(), list(), list(), list()
    with open("data/file.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if 'English' in line and 'timex2norm' in line and 'sgm' in line:
                train_sgm.append(line)
            elif 'English' in line and 'timex2norm' in line and 'apf' in line:
                train_apf.append(line)
        for i in range(40):
            test_apf.append(train_apf[325])
            test_sgm.append(train_sgm[325])
            train_apf.pop(325)
            train_sgm.pop(325)
        for i in range(30):
            rand = random.randint(0, 500)
            dev_apf.append(train_apf[rand])
            dev_sgm.append(train_sgm[rand])
            train_apf.pop(rand)
            train_sgm.pop(rand)
    file_name.extend([train_sgm, train_apf, test_sgm, test_apf, dev_sgm, dev_apf])
    return file_name

def read_file(sgm, apf):
    nlp_path = r'stanford-corenlp-full-2018-10-05'
    tree = ET.parse(apf)
    root = tree.getroot()
    props = {'annotators': 'tokenize'}
    tri_result, arg_result = list(), list()
    with StanfordCoreNLP(nlp_path) as nlp:
        for document in root.findall('document'):
            for event in document.findall('event'):
                event_type = event.attrib.get('SUBTYPE')
                for event_mention in event.findall('event_mention'):
                    # ldc_scope
                    ldc_scope = event_mention.find('ldc_scope')
                    charseq = ldc_scope.find('charseq')
                    ldc_start = int(charseq.attrib.get('START'))
                    sen = charseq.text
                    sens = eval(nlp.annotate(sen, properties=props))
                    sens = sens['tokens']
                    chars, ls = [], []
                    for i in range(len(sens)):
                        chars.append(sens[i]['word'])
                        ls.append(sens[i]['characterOffsetBegin'] + ldc_start)
                    # anchor
                    anchor = event_mention.find('anchor')
                    charseq = anchor.find('charseq')
                    anchor_start = int(charseq.attrib.get('START'))
                    anchor_end = int(charseq.attrib.get('END'))
                    tri = ['0' for i in range(len(chars))]
                    for i in range(len(chars)):
                        if anchor_start <= ls[i] <= anchor_end:
                            tri[i] = event_type
                        if ls[i] > anchor_end + 5:
                            break
                    # argument
                    for argument in event_mention.findall('event_mention_argument'):
                        arg_type = argument.attrib.get('ROLE')
                        extent = argument.find('extent')
                        charseq = extent.find('charseq')
                        arg_start = int(charseq.attrib.get('START'))
                        arg_end = int(charseq.attrib.get('END'))
                        arg = ['0' for i in range(len(chars))]
                        for i in range(len(chars)):
                            if arg_start <= ls[i] <= arg_end:
                                arg[i] = arg_type
                            if ls[i] > arg_end + 5:
                                break
                        tri_result.append([chars, ls, tri, arg])
                        arg_result.append([chars, ls, tri, arg])
        soup = BeautifulSoup(open(sgm), "html5lib")
        text = soup.find(name='text').text
        props = {'annotators': 'ssplit'}
        sens = eval(nlp.annotate(text, properties=props))
        chars, ls = [], []
        for i in range(len(sens['sentences'])):
            chars, ls = [], []
            sen = sens['sentences'][i]
            tokens = sen['tokens']
            tri = ['0' for i in range(len(tokens))]
            arg = ['0' for i in range(len(tokens))]
            for j in range(len(tokens)):
                chars.append(tokens[j]['word'])
                ls.append(tokens[j]['characterOffsetBegin'])
            is_used = False
            for j in range(len(arg_result)):
                if set(arg_result[j][0]) <= set(chars):
                    is_used = True
                    break
            if not is_used:
                tri_result.append([chars, ls, tri, arg])
    return tri_result, arg_result


def prepare_dataset(sgm, apf, sen_len):
    tri_dataset, arg_dataset = list(), list()
    tri_datasets, arg_datasets = list(), list()
    for x in range(len(sgm)):
        print(x)
        tri_data, arg_data = read_file(sgm[x], apf[x])
        tri_dataset += tri_data
        arg_dataset += arg_data
    with open("data/maps.pkl", 'rb') as f:
        char_to_id, id_to_char, _, __ = pickle.load(f)
    for x in range(len(tri_dataset)):
        chars = [char_to_id[x if x in char_to_id else '<UNK>'] for x in tri_dataset[x][0]]
        ls = tri_dataset[x][1]
        tri = get_subtypes(tri_dataset[x][2])
        arg = get_arguroles(tri_dataset[x][3])
        if  2 < len(chars) <= sen_len - 2:
            tri_datasets.append([chars, ls, tri, arg])
    for x in range(len(arg_dataset)):
        chars = [char_to_id[x if x in char_to_id else '<UNK>'] for x in arg_dataset[x][0]]
        ls = arg_dataset[x][1]
        tri = get_subtypes(arg_dataset[x][2])
        arg = get_arguroles(arg_dataset[x][3])
        if 2 < len(chars) <= sen_len - 2:
            arg_datasets.append([chars, ls, tri, arg])
    with open('data/tri.train', 'w') as f:
        for x in range(len(tri_datasets)):
            for y in range(len(tri_datasets[x][0])):
                f.write(str(tri_datasets[x][0][y]))
                f.write(' ')
                f.write(str(tri_datasets[x][1][y]))
                f.write(' ')
                f.write(str(tri_datasets[x][2][y]))
                f.write(' ')
                f.write(str(tri_datasets[x][3][y]))
                f.write('\n')
            f.write('\n')
    with open('data/arg.train', 'w') as f:
        for x in range(len(arg_datasets)):
            for y in range(len(arg_datasets[x][0])):
                f.write(str(arg_datasets[x][0][y]))
                f.write(' ')
                f.write(str(arg_datasets[x][1][y]))
                f.write(' ')
                f.write(str(arg_datasets[x][2][y]))
                f.write(' ')
                f.write(str(arg_datasets[x][3][y]))
                f.write('\n')
            f.write('\n')
    # return tri_datasets, arg_datasets, id_to_char


def get_subtypes(entity_subtype):
    entity_dict = {'0': 0, 'Be-Born': 1, 'Die': 2, 'Marry': 3, 'Divorce': 4, 'Injure': 5, 'Transfer-Ownership': 6,
                   'Transfer-Money': 7, 'Transport': 8, 'Start-Org': 9, 'End-Org': 10, 'Declare-Bankruptcy': 11,
                   'Merge-Org': 12, 'Attack': 13, 'Demonstrate': 14, 'Meet': 15, 'Phone-Write': 16,'Start-Position': 17,
                   'End-Position': 18, 'Nominate': 19, 'Elect': 20, 'Arrest-Jail': 21, 'Release-Parole': 22,
                   'Charge-Indict': 23, 'Trial-Hearing': 24, 'Sue': 25, 'Convict': 26, 'Sentence': 27, 'Fine': 28,
                   'Execute': 29, 'Extradite': 30, 'Acquit': 31, 'Pardon': 32, 'Appeal': 33}
    subtype_featrues = list()
    for w in entity_subtype:
        if w == "O":
            subtype_featrue = 0
        else:
            subtype_featrue = entity_dict[w]
        subtype_featrues.append(subtype_featrue)
    return subtype_featrues


def get_arguroles(roles):
    argument_role_dict = {'0': 0, 'Person': 1, 'Place': 2, 'Buyer': 3, 'Seller': 4, 'Beneficiary': 5, 'Price': 6,
                          'Artifact': 7, 'Origin': 8, 'Destination': 9, 'Giver': 10, 'Recipient': 11, 'Money': 12,
                          'Org': 13, 'Agent': 14, 'Victim': 15, 'Instrument': 16, 'Entity': 17, 'Attacker': 18,
                          'Target': 19, 'Defendant': 20, 'Adjudicator': 21, 'Prosecutor': 22, 'Plaintiff': 23, 'Crime': 24,
                          'Position': 25, 'Sentence': 26, 'Vehicle': 27, 'Time-Within': 28, 'Time-Starting': 29,
                          'Time-Ending': 30, 'Time-Before': 31, 'Time-After': 32, 'Time-Holds': 33, 'Time-At-Beginning': 34,
                          'Time-At-End': 35}
    argument_roles = list()
    for w in roles:
        if w == "O":
            argu_role = 0
        else:
            argu_role = argument_role_dict[w]
        argument_roles.append(argu_role)
    return argument_roles


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
filename = read_filename()
prepare_dataset(filename[0], filename[1], 80)




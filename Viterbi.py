
# coding: utf-8

# In[1]:

from __future__ import division
from collections import Counter
from collections import OrderedDict
import re
import numpy as np
import string
import time
import sys
import random
# In[2]:

def main(train_file, test_file):

    filename = train_file
    test_path = test_file

    # unigram = []

    def create_unigram(filename):
        bigram = []
        bigram_count= {}
        i = 0
        unigram = []
        with open (filename,'r') as training:
            lines = training.readlines()
            start = len(lines)

            for line in lines:
                line = line.split()
                line.append('start')
                for item in line:
                    tag = item.split('/')[-1]
                    # if it's an ambiguous tag we get the second one
                    if "|" in tag:
                        tag = tag.split("|")[-1]

                    unigram.append(tag)
    #         return unigram

        for word in unigram:

            if i < len(unigram)-1:
                bigram.append(unigram[i] + ' '+unigram[i+1])
                i+=1

        new_bigram = [x for x in bigram if not x.endswith("start")]

        for item in new_bigram:       
            if item not in bigram_count:
                bigram_count[item]=1
            else:
                bigram_count[item]+=1

        # tag_frequency, pos_word_prob 
        pos = []
        with open (filename,'r') as training:
            string_r = training.read()

            splited_string = string_r.split()
            for item in splited_string:
                pos.append(item.split('/')[-1]) #Choose the last one when there are multiple slashes.

            tag_frequency = Counter(pos)
            tag_frequency.update({'start':start})

            pos_word_count = Counter(splited_string)
            pos_word_prob = pos_word_count.copy()

            for item in pos_word_count:
                pos_word_prob[item] = pos_word_count[item]/float(tag_frequency[item.split('/')[-1]])

            # bigram_prob
        bigram_prob = {}
        for key,value in bigram_count.items():
            bigram_prob[key]=value/float(tag_frequency[key.split()[0]])

        return unigram, bigram_prob, bigram_count, pos_word_count, pos_word_prob


    # In[3]:


    unigram, bigram_prob, bigram_count, pos_word_count, pos_word_prob = create_unigram(filename)


    # In[4]:


    N1 = 0
    for key, val in pos_word_count.items():
        if val == 1:
            N1 += 1
    turing = N1 / sum(pos_word_count.values())


    # In[5]:


    # unigram, bigram_prob, lines = parse_train()

    all_tags = []
    for key in bigram_prob.keys(): # in order i-1 i
        all_tags.extend(key.split())

    # store unique tags
    tag_list = list(set(all_tags))
    # bigram_prob


    # In[6]:


    # parse the testing file
    test_unigram = []
    # test_path = 'POS.test'
    with open (test_path,'r') as testing:
        test_lines = testing.readlines()
        start = len(test_lines)

        for line in test_lines:
            line = line.split()
            line.append('start')
            for item in line:
                tag = item.split('/')[-1]
                # if it's an ambiguous tag we get the second one
                if "|" in tag:
                    tag = tag.split("|")[-1]

                test_unigram.append(tag)


    # In[7]:


    trans_p = np.zeros(shape=(len(tag_list), len(tag_list)))
    # print(trans_p.shape, len(trans_p))


    # In[8]:




    for col_ind, tag in enumerate(tag_list):
        for row_ind, cond_tag in enumerate(tag_list):

            ti_tim1 = cond_tag + ' ' + tag
            try:
    #             print('trans_p at {},{} is {} w/ val {}'.format(row_ind, col_ind,
    #                                                             ti_tim1, bigram_prob[ti_tim1]))
                trans_p[row_ind, col_ind] = bigram_prob[ti_tim1]
            except:
    #             num_to_share += 1
                    pass
    #             print('trans_p at {},{} is {} w/ val {}'.format(row_ind, col_ind,
    #                                                             ti_tim1, 0))
            


    # In[9]:


    init_state = np.zeros(shape=(len(tag_list),1))


    # In[10]:


    for ind, tag in enumerate(tag_list):
        try:
    #         print('init_state at {}, is {} w/ val {}'.format(ind, 'start' + " "+ tag,
    #                                                          bigram_prob['start' + ' ' + tag]))       
            init_state[ind] = bigram_prob['start' + ' ' + tag]
        except:
            pass


    # In[11]:


    class Decoder(object):
        def __init__(self, initialProb, transProb, obsProb):
            self.N = initialProb.shape[0]
            self.initialProb = initialProb
            self.transProb = transProb
            self.obsProb = obsProb
            assert self.initialProb.shape == (self.N, 1)
            assert self.transProb.shape == (self.N, self.N)
            assert self.obsProb.shape[0] == self.N

        def Obs(self, obs):
            return self.obsProb[:, obs, None]

        def Decode(self, obs):
            trellis = np.zeros((self.N, len(obs)))
            backpt = np.ones((self.N, len(obs)), 'int32') * -1

            # initialization
            # first row of trans matrix time first column of 
            # emission matrix
            trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))

            for t in xrange(1, len(obs)):
                # trellis[:, t-1, None] is previous column of figure 10.9
                # self.Obs(obs[t]) is the t_th column of our p(w_i|t_i) matrix
                trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
                backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
            # termination
            tokens = [trellis[:, -1].argmax()]
            for i in xrange(len(obs)-1, 0, -1):
                tokens.append(backpt[tokens[-1], i])
            return tokens[::-1]


    # # To do/questions
    # - fix p_wi_ti and p_ti_tim1 to include punctuation
    # - create obs matrix and run decoder for each sentence,
    #     - should obs matrix only contain unique words, ie 'janet loves janet' would only have 2 columns?
    #     - for each sentence, get error??
    # - testing? p_wi_ti, p_ti_tim1, obs matrix and transition matrix should all be fixed from using the training data? This means I have to create obs matrix for the whole text in the training data?? So basically, in the training file for sentence one, I map each word to a state and in the text file, I find the state of a given word and use that? if a word is new??

    # In[12]:


    with open(filename, 'r') as train:
        lines = train.readlines()


    # In[13]:



    words = []
    for line in lines:
        for pair in line.split():
            words.append(pair.split('/')[0])


    # In[14]:


    # create ordered word mapping to observation matrix (mainly for testin)
    from collections import OrderedDict

    word_map = OrderedDict()

    ind = 0
    for word in words:
        if word not in word_map:
            word_map[word] = ind
            ind += 1

    word_map['unknown/new word'] = ind


    # In[15]:


    o_mat = np.zeros(shape=(len(tag_list), len(word_map.keys())))
    num_to_share = 0

    for word in word_map.keys():
        for pos in tag_list:
            wi_ti = word + '/' + pos
            if pos_word_prob[wi_ti] == 0:
                num_to_share += 1

    for col_ind, word in enumerate(word_map.keys()):
        for row_ind, pos in enumerate(tag_list):
            wi_ti = word + '/' + pos
            try:
    #             print('obs_mat at {},{} is {} w/ val {}'.format(row_ind, col_map,
    #                                                             wi_ti, pos_word_prob[wi_ti]))
                o_mat[row_ind, col_ind] = pos_word_prob[wi_ti]
            except:
    #             print('obs_mat at {},{} is {} w/ val {}'.format(row_ind, col_ind,
    #                                                             wi_ti, 0))       
                pass
    #             o_mat[row_ind, col_ind] = turing / num_to_share

    o_mat[:, -1] = turing / num_to_share
    decoder = Decoder(initialProb=init_state, transProb=trans_p, obsProb=o_mat)


    # In[16]:


    def run_decoder(decoder, filename=test_path):
        result_list = []
        truth_counter = 0
        num_chances = 0
        num_correct = 0
        result_list = []
        for line in test_lines:
        #     print(line)
            sentence = []
            state = []
            # need to do this 
            for col_ind, txt in enumerate(line.split()):
                word = txt.split('/')[0]
        #         print('word:state is {}:{}'.format(word, word_map[word]))
                try:
                    state.append(word_map[word])
                except: # if we have a new word
        #             print('new', word)
                    state.append(word_map['unknown/new word'])
                sentence.append(word)
            states = decoder.Decode(np.array(state))
            result = np.array(tag_list)[states].tolist()
            resultTagged = zip(sentence,result)
            result_list.append(resultTagged)
        #     print(resultTagged)
        #     print(line)
            for val in result:
                num_chances += 1
                if test_unigram[truth_counter] == 'start':
                    truth_counter += 1
                if val == test_unigram[truth_counter]:
                    num_correct += 1



                truth_counter += 1
        accuracy = num_correct / num_chances
        return accuracy, result_list
    test_accuracy, rslt_list = run_decoder(decoder)


    obs_mat = np.zeros(shape=(len(tag_list), len(word_map.keys())))
    for col_ind, word in enumerate(word_map.keys()):
        for row_ind, pos in enumerate(tag_list):
            wi_ti = word + '/' + pos
            try:
    #             print('obs_mat at {},{} is {} w/ val {}'.format(row_ind, col_map,
    #                                                             wi_ti, pos_word_prob[wi_ti]))
                obs_mat[row_ind, col_ind] = pos_word_count[wi_ti]
            except:
    #             print('obs_mat at {},{} is {} w/ val {}'.format(row_ind, col_ind,
    #                                                             wi_ti, 0))       
                pass
    #             o_mat[row_ind, col_ind] = turing / num_to_share

    # for unknown words, choose from top 5 tags randomly
    pos_mean = obs_mat.sum(axis=1) # should i use the sum or the mean?
    best_five = np.argpartition(pos_mean, -4)[-4:]
    pos_choices = [tag_list[x] for x in best_five]
    # print(pos_choices)
    counter = 0
    num_chances = 0
    num_correct = 0

    for line in test_lines[0:1]:
    #     print(line)
        sentence = []
        state = []
        rslt = []
        # need to do this 
        for col_ind, txt in enumerate(line.split()):
            word = txt.split('/')[0]
            try:
                state.append(word_map[word])
    #             print('word:state is {}:{}'.format(word, word_map[word]))
            except: # if we have a new word
    #             print('word:state is {}:{}'.format(word, word_map['unknown/new word']))
                state.append(word_map['unknown/new word'])
            sentence.append(word)
        for val in state:
            if val == 21162:
                rslt.append(random.choice(pos_choices))
            else:
                rslt.append(tag_list[obs_mat[:,val].argmax()])
        # rsltTagged = zip(sentence, rslt)
        # print(rsltTagged)
        # print(line)
        for val in rslt:
            num_chances += 1
            if test_unigram[counter] == 'start':
                counter += 1
            # print(val, test_unigram[counter])

            if val == test_unigram[counter]:
                num_correct += 1

            counter += 1
    base_accuracy = num_correct / num_chances
    
    return base_accuracy, test_accuracy, rslt_list

if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]
    base_accuracy, test_accuracy, rslt_list = main(train, test)


    print('base accuracy: {} \ntest accuracy: {}'.format(base_accuracy, test_accuracy))


# In[19]:


# with open('POS.test.out', 'w') as outfile:
#     for sent in rslt_list:
#         for pair in sent:
#             outfile.write('%s/%s ' %pair)
#         outfile.write('\n')



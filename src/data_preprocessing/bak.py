#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 15:04 2019-04-25 
@author: haiqinyang

Feature: 

Scenario: 
"""
import queue

# copy from https://github.com/fudannlp16/CWS_Dict
#config=DictConfig
maps_file='checkpoints/maps_msr.pkl'
model_cache='checkpoints/msr2'
UNK='U'
PAD='P'
START='S'
END='E'
TAGB,TAGI,TAGE,TAGS=0,1,2,3

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'

class CWS_Dict:
    def __init__(self,model_name='DictHyperModel'):
        '''
        assert model_name in ['DictConcatModel','DictHyperModel']
        self.word2id,self.id2word,self.dict=cPickle.load(open(maps_file,'rb'))
        self.sess=tf.InteractiveSession()
        self.models = getattr(models,model_name)(vocab_size=len(self.word2id), word_dim=config.word_dim,
                                             hidden_dim=config.hidden_dim,
                                             pad_word=self.word2id[utils_data.PAD], init_embedding=None,
                                             num_classes=config.num_classes, clip=config.clip,
                                             lr=config.lr, l2_reg_lamda=config.l2_reg_lamda,
                                             num_layers=config.num_layers, rnn_cell=config.rnn_cell,
                                             bi_direction=config.bi_direction, hidden_dim2=config.hidden_dim2,
                                             hyper_embedding_size=config.hyper_embed_size)
        self.saver=tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(model_cache)
        self.saver.restore(self.sess,ckpt.model_checkpoint_path)
        '''

    def _findNum(self, sentence):
        '''
        An example
            sent = '迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）\n  ' \
           '中共中央  总书记  、  国家  主席  江  泽民\n' \
           '（  一九九七年  十二月  三十一日  ） \n ' \
           '１２月  ３１日  ，  中共中央  总书记  、  国家  主席  江  泽民  发表  １９９８年  新年  讲话  ' \
           '《  迈向  充满  希望  的  新  世纪  》  。  （  新华社  记者  兰  红光  摄  ）\n' \
           '同胞  们  、  朋友  们  、  女士  们  、  先生  们  ：\n ' \
           '在  １９９８年  来临  之际  ，  我  十分  高兴  地  通过  中央  人民  广播  电台  、' \
           '  中国  国际  广播  电台  和  中央  电视台  ，  向  全国  各族  人民  ，  向  香港  特别  行政区  ' \
           '同胞  、  澳门  和  台湾  同胞  、  海外  侨胞  ，  向  世界  各国  的  朋友  们  ，  致以  诚挚  的  ' \
           '问候  和  良好  的  祝愿  ！  '

            q_num = cws_dict._findNum(sent)
            print(list(q_num.queue))
        '''
        results=queue.Queue()
        for item in re.finditer(rNUM,sentence):
            results.put(item.group())
        return results

    # some problems are here
    def _findEng(self, sentence):
        results=queue.Queue()
        for item in re.finditer(rENG,sentence):
            results.put(item.group())
        return results

    def _preprocess(self,sentence):
        original_sentence=[]
        new_sentence=[]
        num_lists = self._findNum(sentence)
        sentence2=re.sub(rNUM,'0',sentence)
        end_lists = self._findEng(sentence)
        sentence2=re.sub(rENG,'X',sentence2)
        original_sentence=[w for w in sentence2]
        new_sentence=strQ2B(sentence2)
        return original_sentence,new_sentence,num_lists,end_lists

    '''
    def _input_from_line(self,sentence,user_words=None):
        line = sentence
        contexs = utils_data.window(line)
        line_x=[]
        for contex in contexs:
            charx = []
            # contex window
            charx.extend([self.word2id.get(c, self.word2id[utils_data.UNK]) for c in contex])
            # bigram feature
            charx.extend([self.word2id.get(bigram, self.word2id[utils_data.UNK]) for bigram in preprocess.ngram(contex)])
            line_x.append(charx)
        dict_feature=utils_data.tag_sentence(sentence,self.dict,user_words)
        return line_x,dict_feature


    def seg_sentence(self,sentence,user_words=None):
        original_sentence,new_sentence,num_lists,eng_lists=self._preprocess(sentence)
        line_x,dict_feature=self._input_from_line(new_sentence,user_words)
        predict=self.models.predict_step(self.sess,[line_x],[dict_feature])[0]
        seg_result=[]
        word=[]
        for char,tag in zip(original_sentence,predict):
            if char=='0':
                word.append(num_lists.get())
            elif char=='X':
                word.append(eng_lists.get())
            else:
                word.append(char)
            if tag==TAGE or tag==TAGS:
                seg_result.append(''.join(word))
                word=[]
        if len(word)>0:
            seg_result.append(''.join(word))
        return seg_result,predict
    '''
'''end the above'''

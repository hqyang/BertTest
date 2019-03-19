#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 19:10 2019-02-01 
@author: haiqinyang

Feature: 

Scenario: 
"""
from  __future__ import unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import cPickle
#import models
import re
#from config import DictConfig
#import utils_data
#import preprocess
import queue
import pandas as pd
import torch

from src.BERT.modeling import BertConfig
from src.customize_modeling import BertCRFCWS

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
# copy from https://github.com/supercoderhawk/DNN_CWS/blob/master/utils.py
def strQ2B(ustring):
    '''全角转半角'''
    rstring = ''

    for uchar in ustring:
        inside_code = ord(uchar)

        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

    rstring += chr(inside_code)

    return rstring

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False

def escape(text):
    '''html转义'''
    text = (text.replace("&quot;", "\"").replace("&ldquo;", "“").replace("&rdquo;", "”")
          .replace("&middot;", "·").replace("&#8217;", "’").replace("&#8220;", "“")
          .replace("&#8221;", "\”").replace("&#8212;", "——").replace("&hellip;", "…")
          .replace("&#8226;", "·").replace("&#40;", "(").replace("&#41;", ")")
          .replace("&#183;", "·").replace("&amp;", "&").replace("&bull;", "·")
          .replace("&lt;", "<").replace("&#60;", "<").replace("&gt;", ">")
          .replace("&#62;", ">").replace("&nbsp;", " ").replace("&#160;", " ")
          .replace("&tilde;", "~").replace("&mdash;", "—").replace("&copy;", "@")
          .replace("&#169;", "@").replace("♂", "").replace("\r\n|\r", "\n").replace('&nbsp', ' '))
    return text


def read_sogou_report():
    base = 'Reduced/'
    types = os.listdir(base)
    sentences = []
    count = 0
    index = 0
    for type in types:
    # type = 'C000008'
        docs = os.listdir(base + type)

        for doc in docs:
            file = None
            try:
                file = open(base + type + '/' + doc, 'r', encoding='gbk')
                content = escape(strQ2B(file.read())).replace(r'\s', '').replace(r'\n\d+\n', '')
                lines = re.split(r'\n', re.sub(r'[ \t\f]+', r'', content))

                for line in lines:
                    sentences.extend(line.split('。'))
                # break
                file.close()
            except UnicodeDecodeError as e:
                count += 1
                file.close()
                # sentences.append(content)

    return sentences

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
        self.model = getattr(models,model_name)(vocab_size=len(self.word2id), word_dim=config.word_dim,
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

    def _findNum(self,sentence):
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
    def _findEng(self,sentence):
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
        predict=self.model.predict_step(self.sess,[line_x],[dict_feature])[0]
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

def idx_to_tag(tag_to_idx, mode='BMES'):
    idx_to_chunk_tag = {}
    for key, idx in tag_to_idx.items():
        idx_to_chunk_tag[idx] = key

    return idx_to_chunk_tag

def get_Ontonotes(data_dir, type='train'):
    """See base class."""
    df = pd.read_csv(os.path.join(data_dir, type+".tsv"), sep='\t')

    # full_pos (chunk), ner, seg, text
    # need parameter inplace=True
    df.drop(columns=['bert_ner', 'bert_seg', 'full_pos', 'src_ner'], inplace=True)

    # change name to tag for consistently processing
    df.rename(columns={'src_seg': 'label'}, inplace=True)

    return df

def convertList2BMES(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        if not _is_chinese_char(ord(word[0])) or len(word)==1:
            seg_gt = 'S '
        else: # Chinese char and multiple words
            seg_gt = 'B ' + 'M ' * (len(word) - 2) + 'E '

        outStr += seg_gt

        if i==len(rs)-1: # remove the additional space
            outStr = outStr[:-1]

    return outStr

def convertList2BIO(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        if not _is_chinese_char(ord(word[0])) or len(word)==1:
            seg_gt = 'O '
        else: # Chinese char and multiple words
            seg_gt = 'B ' + 'I ' * (len(word) - 1)

        outStr += seg_gt

        if i==len(rs)-1: # remove the additional space
            outStr = outStr[:-1]

    return outStr

def convertList2BIOwithComma(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        if not _is_chinese_char(ord(word[0])) or len(word)==1:
            seg_gt = 'O,'
        else: # Chinese char and multiple words
            seg_gt = 'B,' + 'I,' * (len(word) - 1)

        outStr += seg_gt

        #if i==len(rs)-1: # remove the additional space
        #    outStr = outStr[:-1]

    return outStr

def BMES2BIO(text):
    sOut = text
    sOut = sOut.replace('M', 'I')
    sOut = sOut.replace('E', 'I')
    sOut = sOut.replace('S', 'O')

    return sOut

def space2Comma(text):
    sOut = text
    sOut = sOut.replace(' ', ',')
    if sOut[-1]!=',':
        sOut += ','

    return sOut

def load_model(label_list, args):
    if args.visible_device is not None:
        if isinstance(args.visible_device, int):
            args.visible_device = str(args.visible_device)
        elif isinstance(args.visible_device, (tuple, list)):
            args.visible_device = ','.join([str(_) for _ in args.visible_device])
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.bert_model_dir is not None:
        config_file = os.path.join(args.bert_model_dir, CONFIG_NAME)
        bert_config = BertConfig.from_json_file(config_file)
    else:
        bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.num_hidden_layers>0 and args.num_hidden_layers<bert_config.num_hidden_layers:
        bert_config.num_hidden_layers = args.num_hidden_layers

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if not args.override_output:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            os.system("rm %s" % os.path.join(args.output_dir, '*'))

    model = BertCRFCWS(bert_config, args.vocab_file, args.max_seq_length, len(label_list))

    if args.init_checkpoint is None:
        raise RuntimeError('Evaluating a random initialized model is not supported...!')
    #elif os.path.isdir(args.init_checkpoint):
    #    raise ValueError("init_checkpoint is not a file")
    else:
        weights_path = os.path.join(args.init_checkpoint, WEIGHTS_NAME)

        # main code copy from modeling.py line after 506
        state_dict = torch.load(weights_path)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)

    if args.bert_model is not None:
        weights = torch.load(args.bert_model, map_location='cpu')

        try:
            model.load_state_dict(weights)
        except RuntimeError:
            model.module.load_state_dict(weights)

    return model, device

def export_stat_list(n, param):
    a = param.view(-1)

    return {'name': n, 'max': '{:.6f}'.format(torch.max(a).item()), 'min': '{:.6f}'.format(torch.min(a).item()), \
            'mean': '{:.6f}'.format(torch.mean(a).item()), 'std': '{:.6f}'.format(torch.std(a).item()), \
            'median': '{:.6f}'.format(torch.median(a).item()) }

def save_model(model, fo='tmp.tsv'):
    data_all = [export_stat_list(n, param) for n, param in model.named_parameters()]

    import pandas as pd
    df = pd.DataFrame(data_all)
    # separate with \t
    df.to_csv(fo, sep='\t', encoding='utf-8', index=False,  header=True, \
              columns=['name', 'max', 'min', 'mean', 'median', 'std'])

    print('Finish writing model data to ' + fo + '!')




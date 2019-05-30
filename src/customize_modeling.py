import sys
sys.path.append('../src')
import torch
import torch.nn as nn
import math
from .BERT.modeling import PreTrainedBertModel, BertModel
from .TorchCRF import CRF
from .preprocess import tokenize_text, tokenize_list
from .tokenization import FullTokenizer
import numpy as np
from .utilis import check_english_words, restore_unknown_tokens, append_to_buff, split_text_by_punc
import re
import copy

import pdb


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.qkv = BertQKV(config)
        self.attention_head = BertAttentionHead(config)

    def forward(self, hidden_states, attention_mask):
        q, k, v = self.qkv(hidden_states)
        context_layer = self.attention_head(q, k, v, attention_mask)
        return context_layer

class BertQKV(nn.Module):
    def __init__(self, config):
        super(BertQKV, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        return mixed_query_layer, mixed_key_layer, mixed_value_layer

class BertGroupedQKV(nn.Module):
    def __init__(self, config):
        super(BertGroupedQKV, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = GroupedLinear(config.hidden_size, self.all_head_size, self.num_attention_heads, config)
        self.key = GroupedLinear(config.hidden_size, self.all_head_size, self.num_attention_heads, config)
        self.value = GroupedLinear(config.hidden_size, self.all_head_size, self.num_attention_heads, config)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        return mixed_query_layer, mixed_key_layer, mixed_value_layer    

class BertAttentionHead(nn.Module):
    def __init__(self, config):
        super(BertAttentionHead, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, attention_mask):
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1, config=None):
        super(BertGroupedQKVLinear, self).__init__()
        assert in_features % groups == 0
        assert out_features % groups == 0
        self.groups = groups
        self.in_features = in_features
        self.out_features = out_features
        self.input_group_size = int(self.in_features / self.groups)
        self.output_group_size = int(self.out_features / self.groups)
        self.weight = nn.Parameter(torch.rand(self.groups, self.input_group_size, self.output_group_size))
        self.bias = nn.Parameter(torch.rand(self.groups, 1, self.output_group_size))
        if config is not None:
            self.weight.data.normal_(mean=0.0, std=config.initializer_range)
            self.bias.data.zero_()

    def forward(self, x):
        multihead_input_shape = x.shape[:-1] + (self.groups, self.input_group_size)
        output_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(*multihead_input_shape).permute(0, 2, 1, 3)
        x = torch.matmul(x, self.weight) + self.bias
        x = x.permute(0, 2, 1, 3).contiguous().view(*output_shape)
        return x

    def __str__(self):
        return 'GroupedLinear(in_features=%d, out_features=%d, groups=%d)' \
                     % (self.in_features, self.out_features, self.groups)

    def __repr__(self):
        return self.__str__()


def cut_transformers(model, keep_bottom_layers=6):
    model.bert.encoder.layer = model.bert.encoder.layer[:keep_bottom_layers]
    return model

def use_sparse_qkv(model, keep_bottom_layers=4):
    for layer_index, layer in enumerate(model.bert.encoder.layer):
        if layer_index >= keep_bottom_layers:
            new_attention = customize_modeling.BertSelfAttention(model.bert.config)
            #layer.attention.self.qkv = BertGroupedQKV(model.bert.config)
            layer.attention.self = new_attention
    return model


class BasicBlock(nn.Module):
    def __init__(self, orig_d, proj_size):
        super(BasicBlock, self).__init__()
        self.ffdp = nn.Linear(orig_d, proj_size) # feed forward down projection
        #self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU() # non-linear transformation
        self.ffup = nn.Linear(proj_size, orig_d) # feed forward up projection

    def forward(self, x):
        identity = x

        out = self.ffdp(x)
        out = self.relu(out)
        out = self.ffup(out)

        out += identity
        #out = self.relu(out)

        return out


class BertCRF(PreTrainedBertModel):
    """BERT model with CRF for Sequence Labeling.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output via Conditional Random Field.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_tags = 3

    model = BertCRF(config, num_tags)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_tags=4):
        super(BertCRF, self).__init__(config)
        self.num_tags = num_tags
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Maps the output of BERT into tag space.
        self.hidden2tag = nn.Linear(self.config.hidden_size, num_tags)
        self.classifier = CRF(num_tags, batch_first=True)
        #nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        bert_feats = self.hidden2tag(sequence_output)

        mask = attention_mask.byte()
        if labels is not None:
            loss = -self.classifier(bert_feats, labels, mask)
            return loss
        else:
            raise RuntimeError('Input: labels, is missing!')

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        bert_feats = self.hidden2tag(sequence_output)

        mask = attention_mask.byte()
        loss = bert_feats

        if labels is not None:
            loss = -self.classifier(bert_feats, labels, mask)

        decode_rs = self.classifier.decode(bert_feats, mask)

        return loss, decode_rs


class BertCRFCWS(PreTrainedBertModel):
    """BERT model with CRF for Chinese Word Segmentation.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output via Conditional Random Field.

    Params:
        `device`: the device to set the model.
        `config`: a BertConfig class instance with the configuration to build a new model.
        `vocab_file`: the vocab file for tokenizing the words.
        `max_length`: the maximum length for tokenization.
        `num_tags`: the number of classes for the classifier. Default = 6.
        `batch_size`: the number of mini-batch size for processing the data

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_tags].

    Example usage:
    ```python

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_tags = 6

    model = BertCRFCWS(device, config, vocab_file, max_length, num_tags, batch_size)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, device, config, vocab_file, max_length, num_tags=6, batch_size=64):
        super(BertCRFCWS, self).__init__(config)
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = FullTokenizer(
                vocab_file=vocab_file, do_lower_case=True)
        self.max_length = max_length

        self.num_tags = num_tags
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Maps the output of BERT into tag space.
        self.hidden2tag = nn.Linear(self.config.hidden_size, num_tags)
        self.classifier = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        bert_feats = self.hidden2tag(sequence_output)

        mask = attention_mask.byte()
        if labels is not None:
            loss = -self.classifier(bert_feats, labels, mask)
            return loss
        else:
            raise RuntimeError('Input: labels, is missing!')

    def decode(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        bert_feats = self.hidden2tag(sequence_output)

        mask = attention_mask.byte()

        decode_rs = self.classifier.decode(bert_feats, mask)

        return decode_rs

    def _seg_text(self, words):#->str
        input_ids, segment_ids, input_mask = tokenize_list(words, self.max_length, self.tokenizer)

        input_ids_torch = torch.from_numpy(np.array([input_ids.tolist()])).to(self.device)
        input_mask_torch = torch.from_numpy(np.array([input_mask.tolist()])).to(self.device)

        sequence_output, _ = self.bert(input_ids_torch, input_mask_torch, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        bert_feats = self.hidden2tag(sequence_output)

        input_mask_byte = input_mask_torch.byte()
        #  change input_mask_torch type to byte to avoid RuntimeError: _th_all is not implemented for type torch.LongTensor
        #   in TorchCRF: no_empty_seq_bf = self.batch_first and mask[:, 0].all()
        decode_rs = self.classifier.decode(bert_feats, input_mask_byte)
        tmp_rs = ''.join(str(v) for v in decode_rs[0])

        # tmp_rs[1:-1]: remove the start token and the end token
        decode_output = tmp_rs[1:-1]


        # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
        #  i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}

        # replace the [START] and [END] tokens
        decode_output = decode_output.replace('4', '3') # predict those wrong tokens as a separated word
        decode_output = decode_output.replace('5', '3') #

        return decode_output # a string

    def _seg_wordslist(self, lword):  # ->str
        # lword: list of words (list)
        # input_ids, segment_ids, input_mask = tokenize_list(
        #     words, self.max_length, self.tokenizer)
        input_ids, segment_ids, input_masks = zip(
            *[tokenize_list(w, self.max_length, self.tokenizer) for w in lword])


        input_id_torch = torch.from_numpy(np.array(input_ids)).to(self.device)
        segment_ids_torch = torch.from_numpy(np.array(segment_ids)).to(self.device)
        input_masks_torch = torch.from_numpy(np.array(input_masks)).to(self.device)

        decode_rs = self.decode(input_id_torch, segment_ids_torch, input_masks_torch)

        decode_output_list = []
        for rs in decode_rs:
            tmp_rs = ''.join(str(v) for v in rs)

            # tmp_rs[1:-1]: remove the start token and the end token
            decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
            #  i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}

            # replace the [START] and [END] tokens
            # predict those wrong tokens as a separated word
            # replacing 4 and 5 should not be conducted usually
            decode_output = decode_output.replace('4', '3')
            decode_output = decode_output.replace('5', '3')
            decode_output_list.append(decode_output)

        return decode_output_list  # list of string

    def cut(self, ln, procAll=True):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            model = BertCRFCWS(config, num_tags, vocab_file, max_length)
            output = model.cut(text)
        """
        l = ln.strip('\r\n')
        l = l.strip()

        len_max = self.max_length-2
        wls = self.tokenizer.tokenize(l)

        wls_used = wls
        if not procAll:
            if len(wls_used)>len_max:
                wls = wls[:len_max]
                wls_used = wls

        decode_output = ''
        while len(wls_used) > 0:
            # _seg_wordslist: 02xxx
            decode_output += self._seg_wordslist(wls_used)

            if len(wls_used) > len_max:
                wls_used = wls_used[len_max:]
            else:
                wls_used = []

        result_str = ''
        #pre_word_is_english = False
        cur_word_is_english = False
        for text, tag in zip(wls, decode_output):
            text = text.replace('##', '')
            cur_word_is_english = check_english_words(text)

            if int(tag) > 1: # and (cur_word_is_english)
                # int(tag)>1: tokens of 'E' and 'S'
                # current word is english
                result_str += text + ' '
            else:
                result_str += text

            #pre_word_is_english = cur_word_is_english

        return result_str.strip().split()

    def cutlist(self, input_list):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            model = BertCRFCWS(config, num_tags, vocab_file, max_length)
            output = model.cut(text)
        """
        processed_text_list = []
        merge_index_list = []
        merge_index = 0

        for l_ind, text in enumerate(input_list):
            merge_index_tuple = [merge_index]
            buff = ''

            if isinstance(text, float): continue # process problem of empty line, which is converted to nan

            text_chunk_list = split_text_by_punc(text)
            len_max = self.max_length-2

            for text_chunk in text_chunk_list:
                # if text chunk longer than len_max, split text_chunk
                if len(text_chunk) > len_max:
                    for sub_text_chunk in [
                            text_chunk[i:i+len_max]
                            for i in range(0, len(text_chunk), len_max)]:
                        buff, merge_index = append_to_buff(processed_text_list,
                            buff, sub_text_chunk, len_max, merge_index)
                else:
                    buff, merge_index = append_to_buff(processed_text_list,
                        buff, text_chunk, len_max, merge_index)
            if buff:
                processed_text_list.append(buff)
                merge_index += 1
            merge_index_tuple.append(merge_index)
            merge_index_list.append(merge_index_tuple)
        processed_text_list = [self.tokenizer.tokenize(
            t) for t in processed_text_list]

        decode_output_list = []
        batch_size = self.batch_size
        for p_t_l in [processed_text_list[0+i:batch_size+i] for i in range(0, len(processed_text_list), batch_size)]:
            decode_output_list.extend(self._seg_wordslist(p_t_l))

        #decode_output_list = self._seg_wordslist(processed_text_list)

        # restoring processed_text_list to list of strings
        #processed_text_list = [''.join(char_list) for char_list in processed_text_list]
        result_str_list = []
        for merge_start, merge_end in merge_index_list:
            result_str = ''

            tag = ''.join(decode_output_list[merge_start:merge_end])
            text = []
            for a in processed_text_list[merge_start:merge_end]:
                text.extend(a)

            #text = text.replace('##', '')
            # cur_word_is_english = check_english_words(text)

            for idx in range(len(tag)):
                tt = text[idx]
                tt = tt.replace('##', '')
                ti = tag[idx]
                if int(ti) == 0:  # 'B'
                    result_str += ' ' + tt
                elif int(ti) > 1:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '
                else:
                    result_str += tt

            result_str_list.append(result_str.strip().split())

        return result_str_list
        #result_str += ' '  # separate bcz english issue
        # pre_word_is_english = cur_word_is_english

    def cutlist_noUNK(self, input_list):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            model = BertCRFCWS(config, num_tags, vocab_file, max_length)
            output = model.cutlist_noUNK([text])
        """
        processed_text_list = []
        merge_index_list = []
        merge_index = 0

        for l_ind, text in enumerate(input_list):
            merge_index_tuple = [merge_index]
            buff = ''

            if isinstance(text, float): continue # process problem of empty line, which is converted to nan

            text_chunk_list = split_text_by_punc(text)
            len_max = self.max_length-2

            for text_chunk in text_chunk_list:
                # if text chunk longer than len_max, split text_chunk
                if len(text_chunk) > len_max:
                    for sub_text_chunk in [
                            text_chunk[i:i+len_max]
                            for i in range(0, len(text_chunk), len_max)]:
                        buff, merge_index = append_to_buff(processed_text_list,
                            buff, sub_text_chunk, len_max, merge_index)
                else:
                    buff, merge_index = append_to_buff(processed_text_list,
                        buff, text_chunk, len_max, merge_index)
            if buff:
                processed_text_list.append(buff)
                #original_text_list.append(buff)
                merge_index += 1
            merge_index_tuple.append(merge_index)
            merge_index_list.append(merge_index_tuple)

        original_text_list = processed_text_list
        processed_text_list = [self.tokenizer.tokenize(
            t) for t in processed_text_list]

        decode_output_list = []
        batch_size = self.batch_size
        for p_t_l in [processed_text_list[0+i:batch_size+i] for i in range(0, len(processed_text_list), batch_size)]:
            decode_output_list.extend(self._seg_wordslist(p_t_l))

        #decode_output_list = self._seg_wordslist(processed_text_list)

        # restoring processed_text_list to list of strings
        #processed_text_list = [''.join(char_list) for char_list in processed_text_list]
        result_str_list = []
        for merge_start, merge_end in merge_index_list:
            result_str = ''
            original_str = ''

            tag = ''.join(decode_output_list[merge_start:merge_end])
            text = []
            for a in processed_text_list[merge_start:merge_end]:
                text.extend(a)

            for a in original_text_list[merge_start:merge_end]:
                str_used = ''
                al = re.split('[\n\r]', a)

                for aa in al: str_used += ''.join(aa.strip())

                original_str += str_used

            for idx in range(len(tag)):
                tt = text[idx]
                tt = tt.replace('##', '')
                ti = tag[idx]
                if int(ti) == 0:  # 'B'
                    result_str += ' ' + tt
                elif int(ti) > 1:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '
                else:
                    result_str += tt

            if '[UNK]' in result_str or '[unused' in result_str:
                result_str = restore_unknown_tokens(original_str, result_str)

            result_str_list.append(result_str.strip().split())

        return result_str_list
        #result_str += ' '  # separate bcz english issue
        # pre_word_is_english = cur_word_is_english

    def cut2(self, ln):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            model = BertCRFCWS(config, num_tags, vocab_file, max_length)
            output = model.cut(text)
        """
        ls = split_text_by_punc(ln)
        len_max = self.max_length-2

        result_str = ''
        for wl in ls:
            wls, wls_ori = self.tokenizer.tokenize_with_original(wl)

            wls_used = wls
            if len(wls_used) > len_max:
                wls = wls[:len_max]
                wls_used = wls

            decode_output = ''
            while len(wls_used) > 0:
                # _seg_wordslist: 02xxx
                decode_output += self._seg_text(wls_used)

                if len(wls_used) > len_max:
                    wls_used = wls_used[len_max:]
                else:
                    wls_used = []

            #cur_word_is_english = False
            for text, tag in zip(wls_ori, decode_output):
                text = text.replace('##', '')
                #cur_word_is_english = check_english_words(text)

                if int(tag) == 0: # 'B'
                    result_str += ' ' + text
                elif int(tag) > 1: # and (cur_word_is_english)
                    # int(tag)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += text + ' '
                else:
                    result_str += text

            #result_str += ' ' # separate bcz english issue
                #pre_word_is_english = cur_word_is_english

        if '[UNK]' in result_str:
            original_str = ''.join([text.strip('\r\n').strip() for text in ls])
            result_str = restore_unknown_tokens(original_str, result_str)

        return result_str.strip().split()


class BertSoftMax(PreTrainedBertModel):
    """BERT model with CRF for Sequence Labeling.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output via Conditional Random Field.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_tags = 3

    model = BertSoftMax(config, num_tags)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_tags=4):
        super(BertSoftMax, self).__init__(config)
        self.num_tags = num_tags
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Maps the output of BERT into tag space.
        self.classifier = nn.Linear(self.config.hidden_size, num_tags)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # Utilize the code from BertForTokenClassification
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = logits

        if labels is not None:
            #labels = labels*attention_mask

            #batch_size, seq_length = attention_mask.shape
            #mask = attention_mask.unsqueeze_(-1)
            #mask = mask.expand(batch_size, seq_length, self.num_tags)

            #logits = logits*mask.float()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_tags), labels.view(-1))

        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = logits

        batch_size, seq_length = attention_mask.shape
        if labels is not None:
            #labels = labels*attention_mask

            #mask = attention_mask.unsqueeze_(-1)
            #mask = mask.expand(batch_size, seq_length, self.num_tags)

            #logits = logits*mask.float()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_tags), labels.view(-1))

        best_tags_list = []
        mask = attention_mask.byte()
        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            best_tags = []
            for iseq in range(seq_length):
                if mask[idx, iseq]:
                    _, best_selected_tag = logits[idx, iseq].max(dim=0)
                    best_tags.append(best_selected_tag.item())

            best_tags_list.append(best_tags)

        return loss, best_tags_list


class BertClassifiersCWS(PreTrainedBertModel):
    """BERT model with Multiple Classifiers for Chinese Word Segmentation.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer for Chinese Word Segmentation.

    Params:
        `device`: the device to set the model.
        `config`: a BertConfig class instance with the configuration to build a new model.
        `vocab_file`: the vocab file for tokenizing the words.
        `max_length`: the maximum length for tokenization.
        `num_tags`: the number of classes for the classifier. Default = 6.
        `batch_size`: the number of mini-batch size for processing the data

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_tags].

    Example usage:
    ```python

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertClassifiersCWS(device, config, vocab_file, max_length, num_tags, batch_size)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, device, config, vocab_file, max_length, num_tags=6, batch_size=64):
        super(BertClassifiersCWS, self).__init__(config)
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = FullTokenizer(
                vocab_file=vocab_file, do_lower_case=True)
        self.max_length = max_length

        self.num_tags = num_tags
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Maps the output of BERT into tag space.
        self.classifier = nn.Linear(self.config.hidden_size, num_tags)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # Utilize the code from BertForTokenClassification
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_tags), labels.view(-1))
            return loss
        else:
            return logits

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        bert_feats = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(bert_feats.view(-1, self.num_tags), labels.view(-1))
        else:
            loss = None

        mask = attention_mask.byte()
        batch_size, seq_length = mask.shape

        best_tags_list = []
        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            best_tags = []
            for iseq in range(seq_length):
                if mask[idx, iseq]:
                    _, best_selected_tag = bert_feats[idx, iseq].max(dim=0)
                    best_tags.append(best_selected_tag.item())

            best_tags_list.append(best_tags)

        return loss, best_tags_list

    def _seg_text(self, words):#->str
        input_ids, segment_ids, input_mask = tokenize_list(words, self.max_length, self.tokenizer)

        input_ids_torch = torch.from_numpy(np.array([input_ids.tolist()])).to(self.device)
        input_mask_torch = torch.from_numpy(np.array([input_mask.tolist()])).to(self.device)

        sequence_output, _ = self.bert(input_ids_torch, input_mask_torch, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        bert_feats = self.hidden2tag(sequence_output)

        input_mask_byte = input_mask_torch.byte()
        #  change input_mask_torch type to byte to avoid RuntimeError: _th_all is not implemented for type torch.LongTensor
        #   in TorchCRF: no_empty_seq_bf = self.batch_first and mask[:, 0].all()
        decode_rs = self.classifier.decode(bert_feats, input_mask_byte)
        tmp_rs = ''.join(str(v) for v in decode_rs[0])

        # tmp_rs[1:-1]: remove the start token and the end token
        decode_output = tmp_rs[1:-1]


        # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
        #  i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}

        # replace the [START] and [END] tokens
        decode_output = decode_output.replace('4', '3') # predict those wrong tokens as a separated word
        decode_output = decode_output.replace('5', '3') #

        return decode_output # a string

    def _seg_wordslist(self, lword):  # ->str
        # lword: list of words (list)
        # input_ids, segment_ids, input_mask = tokenize_list(
        #     words, self.max_length, self.tokenizer)
        input_ids, segment_ids, input_masks = zip(
            *[tokenize_list(w, self.max_length, self.tokenizer) for w in lword])


        input_id_torch = torch.from_numpy(np.array(input_ids)).to(self.device)
        segment_ids_torch = torch.from_numpy(np.array(segment_ids)).to(self.device)
        input_masks_torch = torch.from_numpy(np.array(input_masks)).to(self.device)

        decode_rs = self.decode(input_id_torch, segment_ids_torch, input_masks_torch)

        decode_output_list = []
        for rs in decode_rs:
            tmp_rs = ''.join(str(v) for v in rs)

            # tmp_rs[1:-1]: remove the start token and the end token
            decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
            #  i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}

            # replace the [START] and [END] tokens
            # predict those wrong tokens as a separated word
            # replacing 4 and 5 should not be conducted usually
            decode_output = decode_output.replace('4', '3')
            decode_output = decode_output.replace('5', '3')
            decode_output_list.append(decode_output)

        return decode_output_list  # list of string

    def cutlist_noUNK(self, input_list):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            model = BertCRFCWS(config, num_tags, vocab_file, max_length)
            output = model.cutlist_noUNK([text])
        """
        processed_text_list = []
        merge_index_list = []
        merge_index = 0

        for l_ind, text in enumerate(input_list):
            merge_index_tuple = [merge_index]
            buff = ''

            if isinstance(text, float): continue # process problem of empty line, which is converted to nan

            text_chunk_list = split_text_by_punc(text)
            len_max = self.max_length-2

            for text_chunk in text_chunk_list:
                # if text chunk longer than len_max, split text_chunk
                if len(text_chunk) > len_max:
                    for sub_text_chunk in [
                            text_chunk[i:i+len_max]
                            for i in range(0, len(text_chunk), len_max)]:
                        buff, merge_index = append_to_buff(processed_text_list,
                            buff, sub_text_chunk, len_max, merge_index)
                else:
                    buff, merge_index = append_to_buff(processed_text_list,
                        buff, text_chunk, len_max, merge_index)
            if buff:
                processed_text_list.append(buff)
                #original_text_list.append(buff)
                merge_index += 1
            merge_index_tuple.append(merge_index)
            merge_index_list.append(merge_index_tuple)

        original_text_list = processed_text_list
        processed_text_list = [self.tokenizer.tokenize(
            t) for t in processed_text_list]

        decode_output_list = []
        batch_size = self.batch_size
        for p_t_l in [processed_text_list[0+i:batch_size+i] for i in range(0, len(processed_text_list), batch_size)]:
            decode_output_list.extend(self._seg_wordslist(p_t_l))

        #decode_output_list = self._seg_wordslist(processed_text_list)

        # restoring processed_text_list to list of strings
        #processed_text_list = [''.join(char_list) for char_list in processed_text_list]
        result_str_list = []
        for merge_start, merge_end in merge_index_list:
            result_str = ''
            original_str = ''

            tag = ''.join(decode_output_list[merge_start:merge_end])
            text = []
            for a in processed_text_list[merge_start:merge_end]:
                text.extend(a)

            for a in original_text_list[merge_start:merge_end]:
                str_used = ''
                al = re.split('[\n\r]', a)

                for aa in al: str_used += ''.join(aa.strip())

                original_str += str_used

            for idx in range(len(tag)):
                tt = text[idx]
                tt = tt.replace('##', '')
                ti = tag[idx]
                if int(ti) == 0:  # 'B'
                    result_str += ' ' + tt
                elif int(ti) > 1:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '
                else:
                    result_str += tt

            if '[UNK]' in result_str or '[unused' in result_str:
                result_str = restore_unknown_tokens(original_str, result_str)

            result_str_list.append(result_str.strip().split())

        return result_str_list
        #result_str += ' '  # separate bcz english issue
        # pre_word_is_english = cur_word_is_english


class MultiHeadMultiLayerAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadMultiLayerAttention, self).__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.dense_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.layer_k = nn.ModuleList([copy.deepcopy(self.dense_k) for _ in range(config.num_hidden_layers)])
        self.dense_a = nn.Linear(config.num_hidden_layers, 1)
        self.layer_a = nn.ModuleList([copy.deepcopy(self.dense_a) for _ in range(config.num_hidden_layers)])
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # do not use truncated_normal as TF for initialization            #
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, hidden_states):
        all_encoder_layers = []
        attention_weights = []
        for i in range(self.num_hidden_layers):
            hidden_state_i = hidden_states[i]
            hidden_state_i = self.layer_k[i](hidden_state_i)
            all_encoder_layers.append(hidden_state_i)

            hidden_state_i = self.activation(hidden_state_i)
            attention_weight = self.layer_a[i](hidden_state_i)
            attention_weights.append()
        return


class BertVariant(PreTrainedBertModel):
    """Apply BERT fixed features with BiLSTM and CRF for Sequence Labeling.

    model = BertVariant(config, num_tags)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_tags=4, method=None, fclassifier=None):
        super(BertVariant, self).__init__(config)
        self.num_tags = num_tags
        self.method = method
        self.fclassifier = fclassifier
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        if method == 'fine_tune':
            last_hidden_size = self.config.hidden_size
        elif method == 'cat_last4':
            self.biLSTM = nn.LSTM(input_size=self.config.hidden_size*4,
                                  hidden_size=self.config.hidden_size,
                                  num_layers=2, batch_first=True,
                                  dropout=0, bidirectional=True)
            last_hidden_size = self.config.hidden_size*2
        elif method in ['last_layer', 'sum_last4', 'sum_all', 'cat_last4']:
            self.biLSTM = nn.LSTM(input_size=self.config.hidden_size,
                                  hidden_size=self.config.hidden_size,
                                  num_layers=2, batch_first=True,
                                  dropout=0, bidirectional=True)
            last_hidden_size = self.config.hidden_size*2
        elif self.method == 'MHMLA':
            self.MHMLA = MultiHeadMultiLayerAttention(config)

        # Maps the output of BERT into tag space.
        self.hidden2tag = nn.Linear(last_hidden_size, num_tags)

        if self.fclassifier == 'CRF':
            self.classifier = CRF(num_tags, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        logits = self._compute_bert_feats(input_ids, token_type_ids, attention_mask)

        mask = attention_mask.byte()
        if labels is None:
            raise RuntimeError('Input: labels, is missing!')
        else:
            loss = self._compute_loss(logits, mask, labels)
        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        logits = self._compute_bert_feats(input_ids, token_type_ids, attention_mask)

        loss = logits

        mask = attention_mask.byte()
        if labels is not None:
            loss = self._compute_loss(logits, mask, labels)

        if self.fclassifier == 'CRF':
            best_tags_list = self.classifier.decode(logits, mask)
        elif self.fclassifier == 'Softmax':
            best_tags_list = self._decode_Softmax(logits, mask)

        return loss, best_tags_list

    def _compute_bert_feats(self, input_ids, token_type_ids=None, attention_mask=None):
        if self.method in ['last_layer', 'fine_tune']:
            output_all_encoded_layers = False
        else: # sum_last4, sum_all, cat_last4, 'MHMLA'
            output_all_encoded_layers = True

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=output_all_encoded_layers)

        if self.method == 'sum_last4':
            feat_used = self.dropout(sequence_output[-1])
            for l in range(-2, -5, -1):
                feat_used += self.dropout(sequence_output[l])
        elif self.method == 'sum_all':
            feat_used = self.dropout(sequence_output[-1])
            for l in range(-2, -13, -1):
                feat_used += self.dropout(sequence_output[l])
        elif self.method == 'cat_last4':
            feat_used = self.dropout(sequence_output[-4])
            for l in range(-3, 0):
                feat_used = torch.cat((feat_used, self.dropout(sequence_output[l])), 2)
        elif self.method in ['last_layer', 'fine_tune']:
            feat_used = sequence_output
            feat_used = self.dropout(feat_used)
        elif self.method == 'MHMLA':
            feat_used = self.MHMLA(sequence_output)

        if self.method in ['sum_last4', 'sum_all', 'cat_last4', 'last_layer']:
            feat_used, _ = self.biLSTM(feat_used)

        bert_feats = self.hidden2tag(feat_used)

        return bert_feats

    def _compute_loss(self, logits, mask, labels):
        # mask is a ByteTensor

        if self.fclassifier == 'Softmax':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_tags), labels.view(-1))
        elif self.fclassifier == 'CRF':
            loss = -self.classifier(logits, labels, mask)

        return loss

    def _decode_Softmax(self, logits, mask):
        # mask is a ByteTensor

        batch_size, _ = mask.shape

        _, best_selected_tag = logits.max(dim=2)

        best_tags_list = []
        for n in range(batch_size):
            selected_tag = torch.masked_select(best_selected_tag[n, :], mask[n, :])
            best_tags_list.append(selected_tag.tolist())

        return best_tags_list


class BertCWS(PreTrainedBertModel):
    """BERT model with CRF for Chinese Word Segmentation.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output via Conditional Random Field.

    Params:
        `device`: the device to set the model.
        `config`: a BertConfig class instance with the configuration to build a new model.
        `vocab_file`: the vocab file for tokenizing the words.
        `max_length`: the maximum length for tokenization.
        `num_tags`: the number of classes for the classifier. Default = 6.
        `batch_size`: the number of mini-batch size for processing the data
        'fclassifier': the type of classifier in the final stage, currently I use CRF or Softmax

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_tags].

    Example usage:
    ```python

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_tags = 6

    model = BertCWS(device, config, vocab_file, max_length, num_tags, fclassifier, batch_size)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, device, config, vocab_file, max_length, num_tags=6, batch_size=64, fclassifier='Softmax', method='fine_tune'):
        super(BertCWS, self).__init__(config)
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = FullTokenizer(
                vocab_file=vocab_file, do_lower_case=True)
        self.max_length = max_length

        self.num_tags = num_tags
        self.fclassifier = fclassifier
        self.method = method
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        if method == 'fine_tune':
            last_hidden_size = self.config.hidden_size
        elif method == 'cat_last4':
            self.biLSTM = nn.LSTM(input_size=self.config.hidden_size*4,
                                  hidden_size=self.config.hidden_size,
                                  num_layers=2, batch_first=True,
                                  dropout=0, bidirectional=True)
            last_hidden_size = self.config.hidden_size*2
        elif method in ['last_layer', 'sum_last4', 'sum_all', 'cat_last4']:
            self.biLSTM = nn.LSTM(input_size=self.config.hidden_size,
                                  hidden_size=self.config.hidden_size,
                                  num_layers=2, batch_first=True,
                                  dropout=0, bidirectional=True)
            last_hidden_size = self.config.hidden_size*2
        elif self.method == 'MHMLA':
            self.MHMLA = MultiHeadMultiLayerAttention(config)

        # Maps the output of BERT into tag space.
        self.hidden2tag = nn.Linear(self.config.hidden_size, num_tags)

        if self.fclassifier == 'CRF':
            self.classifier = CRF(num_tags, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        logits = self._compute_bert_feats(input_ids, token_type_ids, attention_mask)

        mask = attention_mask.byte()
        if labels is None:
            raise RuntimeError('Input: labels, is missing!')
        else:
            loss = self._compute_loss(logits, mask, labels)
        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        logits = self._compute_bert_feats(input_ids, token_type_ids, attention_mask)

        loss = logits

        mask = attention_mask.byte()
        if labels is not None:
            loss = self._compute_loss(logits, mask, labels)

        if self.fclassifier == 'CRF':
            best_tags_list = self.classifier.decode(logits, mask)
        elif self.fclassifier == 'Softmax':
            best_tags_list = self._decode_Softmax(logits, mask)

        return loss, best_tags_list

    def _compute_bert_feats(self, input_ids, token_type_ids=None, attention_mask=None):
        if self.method in ['last_layer', 'fine_tune']:
            output_all_encoded_layers = False
        else: # , 'MHMLA'
            output_all_encoded_layers = True

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=output_all_encoded_layers)

        if self.method == 'sum_last4':
            feat_used = self.dropout(sequence_output[-1])
            for l in range(-2, -5, -1):
                feat_used += self.dropout(sequence_output[l])
        elif self.method == 'sum_all':
            feat_used = self.dropout(sequence_output[-1])
            for l in range(-2, -13, -1):
                feat_used += self.dropout(sequence_output[l])
        elif self.method == 'cat_last4':
            feat_used = self.dropout(sequence_output[-4])
            for l in range(-3, 0):
                feat_used = torch.cat((feat_used, self.dropout(sequence_output[l])), 2)
        elif self.method in ['last_layer', 'fine_tune']:
            feat_used = sequence_output
            feat_used = self.dropout(feat_used)
        elif self.method == 'MHMLA':
            feat_used = self.MHMLA(sequence_output)

        if self.method in ['sum_last4', 'sum_all', 'cat_last4', 'last_layer']:
            feat_used, _ = self.biLSTM(feat_used)

        bert_feats = self.hidden2tag(feat_used)

        return bert_feats

    def _compute_loss(self, logits, mask, labels):
        # mask is a ByteTensor

        if self.fclassifier == 'Softmax':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_tags), labels.view(-1), mask.view(-1))
        elif self.fclassifier == 'CRF':
            loss = -self.classifier(logits, labels, mask)

        return loss

    def _decode_Softmax(self, logits, mask):
        # mask is a ByteTensor

        batch_size, _ = mask.shape

        _, best_selected_tag = logits.max(dim=2)

        best_tags_list = []
        for n in range(batch_size):
            selected_tag = torch.masked_select(best_selected_tag[n, :], mask[n, :])
            best_tags_list.append(selected_tag.tolist())

        return best_tags_list

    def _seg_wordslist(self, lword):  # ->str
        # lword: list of words (list)
        # input_ids, segment_ids, input_mask = tokenize_list(
        #     words, self.max_length, self.tokenizer)
        input_ids, segment_ids, input_masks = zip(
            *[tokenize_list(w, self.max_length, self.tokenizer) for w in lword])

        input_id_torch = torch.from_numpy(np.array(input_ids)).to(self.device)
        segment_ids_torch = torch.from_numpy(np.array(segment_ids)).to(self.device)
        input_masks_torch = torch.from_numpy(np.array(input_masks)).to(self.device)

        _, decode_rs = self.decode(input_id_torch, segment_ids_torch, input_masks_torch)

        decode_output_list = []
        for rs in decode_rs:
            tmp_rs = ''.join(str(v) for v in rs)

            # tmp_rs[1:-1]: remove the start token and the end token
            decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
            #  i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}

            # replace the [START] and [END] tokens
            # predict those wrong tokens as a separated word
            # replacing 4 and 5 should not be conducted usually
            decode_output = decode_output.replace('4', '3')
            decode_output = decode_output.replace('5', '3')
            decode_output_list.append(decode_output)

        return decode_output_list  # list of string

    def cutlist_noUNK(self, input_list):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            model = BertCWS(config, num_tags, vocab_file, max_length)
            output = model.cutlist_noUNK([text])
        """
        processed_text_list = []
        merge_index_list = []
        merge_index = 0

        for l_ind, text in enumerate(input_list):
            merge_index_tuple = [merge_index]
            buff = ''

            if isinstance(text, float): continue # process problem of empty line, which is converted to nan

            text_chunk_list = split_text_by_punc(text)
            len_max = self.max_length-2

            for text_chunk in text_chunk_list:
                # if text chunk longer than len_max, split text_chunk
                if len(text_chunk) > len_max:
                    for sub_text_chunk in [
                            text_chunk[i:i+len_max]
                            for i in range(0, len(text_chunk), len_max)]:
                        buff, merge_index = append_to_buff(processed_text_list,
                            buff, sub_text_chunk, len_max, merge_index)
                else:
                    buff, merge_index = append_to_buff(processed_text_list,
                        buff, text_chunk, len_max, merge_index)
            if buff:
                processed_text_list.append(buff)
                #original_text_list.append(buff)
                merge_index += 1
            merge_index_tuple.append(merge_index)
            merge_index_list.append(merge_index_tuple)

        original_text_list = processed_text_list
        processed_text_list = [self.tokenizer.tokenize(
            t) for t in processed_text_list]

        decode_output_list = []
        batch_size = self.batch_size
        for p_t_l in [processed_text_list[0+i:batch_size+i] for i in range(0, len(processed_text_list), batch_size)]:
            decode_output_list.extend(self._seg_wordslist(p_t_l))

        # restoring processed_text_list to list of strings
        #processed_text_list = [''.join(char_list) for char_list in processed_text_list]
        result_str_list = []
        for merge_start, merge_end in merge_index_list:
            result_str = ''
            original_str = ''

            tag = ''.join(decode_output_list[merge_start:merge_end])
            text = []
            for a in processed_text_list[merge_start:merge_end]:
                text.extend(a)

            for a in original_text_list[merge_start:merge_end]:
                str_used = ''
                al = re.split('[\n\r]', a)

                for aa in al: str_used += ''.join(aa.strip())

                original_str += str_used

            for idx in range(len(tag)):
                tt = text[idx]
                tt = tt.replace('##', '')
                ti = tag[idx]

                try:
                    int(ti)
                except ValueError:
                    print(ti + '\n')      # or whatever
                    print(tag)

                if int(ti) == 0:  # 'B'
                    result_str += ' ' + tt
                elif int(ti) > 1:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '
                else:
                    result_str += tt

            if '[UNK]' in result_str or '[unused' in result_str:
                result_str = restore_unknown_tokens(original_str, result_str)

            result_str_list.append(result_str.strip().split())

        return result_str_list

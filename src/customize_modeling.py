import torch
import torch.nn as nn
import math
from .BERT.modeling import PreTrainedBertModel, BertModel, BertLayerNorm, BertEncoder, BertPooler
from .TorchCRF import CRF
from .preprocess import read_dict, tokenize_list, define_words_set, tokenize_list_with_cand_indexes_lang_status, \
            tokenize_list_with_cand_indexes_lang_status_dict_vec
from .tokenization import FullTokenizer
from .BERT.tokenization import BertTokenizer
import numpy as np
from .utilis import unpackTuple, restore_unknown_tokens, restore_unknown_tokens_with_pos, append_to_buff, \
    split_text_by_punc, extract_pos, restore_unknown_tokens_without_unused_with_pos
import re
import copy
from .config import segType, posType, MAX_GRAM_LEN
import time

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
            #layer.attention.self.qkv = BertGroupedQKV(models.bert.config)
            layer.attention.self = new_attention
    return model


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

    models = BertVariant(config, num_tags)
    logits = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_tags=4, method='fine_tune', fclassifier='Softmax'):
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
        elif method in ['last_layer', 'sum_last4', 'sum_all']:
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


class BertCWS(BertVariant):
    """BERT models with CRF for Chinese Word Segmentation.
    This module is composed of the BERT models with a linear layer on top of
    the pooled output via Conditional Random Field.

    Params:
        `device`: the device to set the models.
        `config`: a BertConfig class instance with the configuration to build a new models.
        `vocab_file`: the models file for tokenizing the words.
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

    models = BertCWS(device, config, vocab_file, max_length, num_tags, fclassifier, batch_size)
    logits = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, device, config, vocab_file, max_length, num_tags=6, batch_size=64, fclassifier='Softmax', method='fine_tune'):
        super(BertCWS, self).__init__(config)
        BertVariant.__init__(self, config, num_tags=num_tags, method=method, fclassifier=fclassifier)

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

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
            decode_output = ''.join(str(v) for v in rs[1:-1])

            # tmp_rs[1:-1]: remove the start token and the end token
            #decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
            #  i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}
            #  i.e., BMES_idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'M', 4: 'E', 5: 'S'}

            # replace the [START] and [END] tokens
            # predict those wrong tokens as a separated word
            # replacing 4 and 5 should not be conducted usually
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[START]']), str(segType.BMES_label_map['S']))
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[END]']), str(segType.BMES_label_map['S']))
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

            models = BertCWS(config, num_tags, vocab_file, max_length)
            output = models.cutlist_noUNK([text])
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

                if int(ti) == segType.BMES_label_map['B']:  # 'B'
                    result_str += ' ' + tt
                elif int(ti) > segType.BMES_label_map['M']:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '
                else:
                    result_str += tt

            if UNK_TOKEN in result_str or '[unused' in result_str:
                result_str = restore_unknown_tokens(original_str, result_str)

            result_str_list.append(result_str.strip().split())

        return result_str_list


class BertVariantCWSPOS(PreTrainedBertModel):
    """Apply BERT for Sequence Labeling on Chinese Word Segmentation and Part-of-Speech.

    models = BertVariantCWSPOS(config, num_tags)
    logits = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_CWStags=6, num_POStags=108, method='fine_tune', fclassifier='Softmax'):
        super(BertVariantCWSPOS, self).__init__(config)
        self.num_CWStags = num_CWStags
        self.num_POStags = num_POStags
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
        elif method in ['last_layer', 'sum_last4', 'sum_all']:
            self.biLSTM = nn.LSTM(input_size=self.config.hidden_size,
                                  hidden_size=self.config.hidden_size,
                                  num_layers=2, batch_first=True,
                                  dropout=0, bidirectional=True)
            last_hidden_size = self.config.hidden_size*2
        elif self.method == 'MHMLA':
            self.MHMLA = MultiHeadMultiLayerAttention(config)

        # Maps the output of BERT into tag space.
        self.hidden2CWStag = nn.Linear(last_hidden_size, num_CWStags)
        self.hidden2POStag = nn.Linear(last_hidden_size, num_POStags)

        if self.fclassifier == 'CRF':
            self.CWSclassifier = CRF(num_CWStags, batch_first=True)
            self.POSclassifier = CRF(num_POStags, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels_CWS=None, labels_POS=None):
        mask = attention_mask.byte()
        loss = 1e10

        if labels_CWS is None and labels_POS is None:
            raise RuntimeError('Input: labels_CWS or labels_POS is missing!')
        else:
            feat_used = self._compute_bert_feats(input_ids, token_type_ids, attention_mask)
            cws_logits = self.hidden2CWStag(feat_used)

            if labels_CWS is not None:
                loss = self._compute_loss(cws_logits, mask, labels_CWS, 'CWS')

            if labels_POS is not None:
                pos_logits = self.hidden2POStag(feat_used)
                loss += self._compute_loss(pos_logits, mask, labels_POS, 'POS')

        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, labels_CWS=None, labels_POS=None):
        cws_loss = 1e10
        pos_loss = 1e10

        mask = attention_mask.byte()
        feat_used = self._compute_bert_feats(input_ids, token_type_ids, attention_mask)

        cws_logits = self.hidden2CWStag(feat_used)
        pos_logits = self.hidden2POStag(feat_used)

        if labels_CWS is not None:
            cws_loss = self._compute_loss(cws_logits, mask, labels_CWS, 'CWS')

        if labels_POS is not None:
            pos_loss = self._compute_loss(pos_logits, mask, labels_POS, 'POS')

        if self.fclassifier == 'CRF':
            best_cws_tags_list = self.classifier.decode(cws_logits, mask)
            best_pos_tags_list = self.classifier.decode(pos_logits, mask)
        elif self.fclassifier == 'Softmax':
            best_cws_tags_list = self._decode_Softmax(cws_logits, mask)
            best_pos_tags_list = self._decode_Softmax(pos_logits, mask)

        return cws_loss, pos_loss, best_cws_tags_list, best_pos_tags_list

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

        #bert_feats = self.hidden2tag(feat_used)

        return feat_used

    def _compute_loss(self, logits, mask, labels, task):
        # mask is a ByteTensor

        if self.fclassifier == 'Softmax':
            loss_fct = nn.CrossEntropyLoss()

            if task == 'CWS':
                num_tags = self.num_CWStags
            elif task == 'POS':
                num_tags = self.num_POStags

            loss = loss_fct(logits.view(-1, num_tags), labels.view(-1))
        elif self.fclassifier == 'CRF':
            if task == 'CWS':
                loss = -self.CWSclassifier(logits, labels, mask)
            elif task == 'POS':
                loss = -self.POSclassifier(logits, labels, mask)

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


class BertCWSPOS(BertVariantCWSPOS):
    """Apply BERT for Sequence Labeling on Chinese Word Segmentation and Part-of-Speech.

    models = BertCWSPOS(device, config, vocab_file, max_length, num_CWStags=6, num_POStags=110, batch_size=64, fclassifier='Softmax', method='fine_tune')
    logits = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, device, config, vocab_file, max_length, num_CWStags=6, num_POStags=110, batch_size=64, fclassifier='Softmax', pclassifier='CRF', method='fine_tune'):
        super(BertCWSPOS, self).__init__(config)
        BertVariantCWSPOS.__init__(self, config, num_CWStags=num_CWStags, num_POStags=num_POStags, method=method, fclassifier=fclassifier)

        self.device = device
        self.batch_size = batch_size
        self.tokenizer = FullTokenizer(
                vocab_file=vocab_file, do_lower_case=True)
        self.max_length = max_length


    def _seg_wordslist(self, lword):  # ->str
        # lword: list of words (list)
        # input_ids, segment_ids, input_mask = tokenize_list(
        #     words, self.max_length, self.tokenizer)
        input_ids, segment_ids, input_masks = zip(
            *[tokenize_list(w, self.max_length, self.tokenizer) for w in lword])
            #*[tokenize_list_no_seg(w, self.max_length, self.tokenizer) for w in lword])

        input_id_torch = torch.from_numpy(np.array(input_ids)).to(self.device)
        segment_ids_torch = torch.from_numpy(np.array(segment_ids)).to(self.device)
        input_masks_torch = torch.from_numpy(np.array(input_masks)).to(self.device)

        _, _, best_cws_tags_list, best_pos_tags_list = self.decode(input_id_torch, segment_ids_torch, input_masks_torch)

        cws_output_list = []
        for rs in best_cws_tags_list:
            cws_decode_output = ''.join(str(v) for v in rs[1:-1]) #

            # tmp_rs[1:-1]: remove the tokens, [START] and [END]
            #decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
            # i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}
            # i.e., BMES_idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'M', 4: 'E', 5: 'S'}

            # replace the [START] and [END] tokens
            # predict those wrong tokens as a separated word
            # replacing 0 and 1 should not be conducted usually
            cws_decode_output = cws_decode_output.replace(str(segType.BMES_label_map['[START]']), str(segType.BMES_label_map['S']))
            cws_decode_output = cws_decode_output.replace(str(segType.BMES_label_map['[END]']), str(segType.BMES_label_map['S']))

            cws_output_list.append(cws_decode_output)

        pos_output_list = []
        for rs in best_pos_tags_list:
            pos_decode_output = ' '.join(posType.POS_label_map[(v-2)//3] if v > 2 else posType.POS_label_map[35] for v in rs[1:-1]) #

            # tmp_rs[1:-1]: remove the tokens, [START] and [END]
            #decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens in POSType.BIO_idx_to_label_map

            # replace the [START] and [END] tokens ??
            # predict those wrong tokens as a separated word
            # replacing 0 and 1 should not be conducted usually
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[START]']), str(segType.BMES_label_map['S']))
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[END]']), str(segType.BMES_label_map['S']))

            pos_output_list.append(pos_decode_output)

        return cws_output_list, pos_output_list  # list of string

    def cutlist_noUNK(self, input_list):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            models = BertCWS(config, num_tags, vocab_file, max_length)
            output = models.cutlist_noUNK([text])
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

        cws_output_list = []
        pos_output_list = []
        tmp_pos_list = []

        batch_size = self.batch_size
        for p_t_l in [processed_text_list[0+i:batch_size+i] for i in range(0, len(processed_text_list), batch_size)]:
            cws_output, pos_output = self._seg_wordslist(p_t_l)
            cws_output_list.extend(cws_output)
            pos_output_list.extend(pos_output)

        # restoring processed_text_list to list of strings
        #processed_text_list = [''.join(char_list) for char_list in processed_text_list]
        result_str_list = []

        for merge_start, merge_end in merge_index_list:
            result_str = ''
            original_str = ''
            result_pos = '' # storing pos results

            cws_tag = ''.join(cws_output_list[merge_start:merge_end])
            pos_tag = ' '.join(pos_output_list[merge_start:merge_end]).split()

            text = []
            for a in processed_text_list[merge_start:merge_end]:
                text.extend(a)

            for a in original_text_list[merge_start:merge_end]:
                str_used = ''
                al = re.split('[\n\r]', a)

                for aa in al: str_used += ''.join(aa.strip())

                original_str += str_used

            tmp_pos = []
            seg_start = False
            for idx in range(len(cws_tag)):
                tt = text[idx]
                tt = tt.replace('##', '')
                ti = cws_tag[idx]
                pos_tag_i = pos_tag[idx]

                try:
                    int(ti)
                except ValueError:
                    print(ti + '\n')      # or whatever
                    print(cws_tag)

                int_ti = int(ti)
                if int_ti == segType.BMES_label_map['B']:  # 'B'
                    result_str += ' ' + tt

                    if not seg_start:
                        seg_start = True

                    if tmp_pos != []:
                        tmp_pos_list.append(tmp_pos)

                    result_pos += pos_tag_i + ' '
                    tmp_pos = [pos_tag_i]
                elif int_ti > segType.BMES_label_map['M']:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '

                    #if int_ti == segType.BMES_label_map['S']:
                    #    result_pos += pos_tag_i + ' '
                    #    tmp_pos = []

                    #    tmp_pos_list.append([pos_tag_i])
                    #else:
                    if tmp_pos == []:
                        result_pos += pos_tag_i + ' '

                    tmp_pos.extend([pos_tag_i])
                    tmp_pos_list.append(tmp_pos)
                    tmp_pos = []

                    seg_start = False
                else:
                    result_str += tt
                    tmp_pos.append(pos_tag_i)
                    if not seg_start:
                        seg_start = True
                        result_pos += pos_tag_i + ' '

            result_pos_str = extract_pos(tmp_pos_list)

            if '[UNK]' in result_str or '[unused' in result_str:
                seg_ls, pos_ls = restore_unknown_tokens_with_pos(original_str, result_str, result_pos)
            else:
                seg_ls = result_str.strip().split()
                pos_ls = result_pos.strip().split()

            #seg_ls = result_str_rev.strip().split()
            #pos_ls = result_pos_rev.strip().split()
            assert(len(seg_ls)==len(pos_ls))

            rs = []
            for i in range(len(seg_ls)):
                rs.append(seg_ls[i] + ' / ' + pos_ls[i])

            result_str_list.append(rs)

        return result_str_list


class BertMLEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings for multilinguisticss
    """
    def __init__(self, config, update_method='mean', speedup=True):
        super(BertMLEmbeddings, self).__init__()
        self.hidden_size = config.hidden_size
        self.update_method = update_method
        self.speedup = speedup
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cand_indexes=None, token_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if self.speedup:
            words_embeddings = self.extract_embedding_speed(token_ids, attention_mask, cand_indexes)
        else:
            words_embeddings = self.extract_embedding(token_ids, attention_mask, cand_indexes)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

#    def extract_embedding_ij(self, token_id_i, max_chunk_per_word):
#        words_embeddings = torch.where(torch.isnan(words_embeddings), torch.zeros_like(words_embeddings), words_embeddings)
#        cand_mask = token_id.ge(1)
#        return word_embedding_ij

#    def extract_embedding_i(self, token_id_i, max_seq_len, max_chunk_per_word):
#        # max_seq_len * max_chunk_per_word
#        for j in range(max_seq_len):
#            words_embeddings_i[j] = self.extract_embedding_ij(token_id_i[j], max_chunk_per_word)
#        return words_embeddings

    def extract_embedding_speed(self, token_ids, attention_mask=None, cand_indexes=None):
        if token_ids is None: # cand_indexes is None and
            raise RuntimeError('Input: cand_indexes or token_ids should not be None!')

        batch_size, max_seq_len, max_chunk_per_word = token_ids.shape
        cand_mask = token_ids.ge(1)
        token_ids_2d = token_ids.view(batch_size*max_seq_len*max_chunk_per_word, 1)
        words_embeddings = self.word_embeddings(token_ids_2d)
        words_embeddings = torch.where(torch.isnan(words_embeddings), torch.zeros_like(words_embeddings), words_embeddings)

        # [batch_size, max_seq_len, max_chunk_per_word, hidden_size]
        words_embeddings = words_embeddings.view(batch_size, max_seq_len, max_chunk_per_word, -1)

        # [batch_size, max_seq_len, hidden_size]
        words_embeddings = torch.sum(
            words_embeddings*cand_mask.unsqueeze(3).float(), dim=2) / torch.sum(cand_mask, dim=2).unsqueeze(2).float()

        words_embeddings = torch.where(torch.isnan(words_embeddings), torch.zeros_like(words_embeddings), words_embeddings)

        return words_embeddings

    def extract_embedding(self, token_ids, attention_mask=None, cand_indexes=None):
        if token_ids is None: # cand_indexes is None and
            raise RuntimeError('Input: cand_indexes or token_ids should not be None!')

        batch_size, max_seq_len, max_chunk_per_word = token_ids.size()
        words_embeddings = torch.zeros(batch_size, max_seq_len, self.hidden_size).to(token_ids.device)

        for i in range(batch_size):
            seq_len = torch.sum(attention_mask[i])
            for j in range(seq_len):
                token_idx = token_ids[i][j]
                cand_mask = token_idx.ge(1)
                words_embedding_ij = self.word_embeddings(token_idx[cand_mask])
                words_embeddings[i][j] = torch.sum(words_embedding_ij, dim=0)/torch.sum(cand_mask)

        # word_embedding has shape [batch_size*max_seq_len*max_chunk_per_word, hidden_size]
        return words_embeddings

    def update_embedding_speed(self, word_embedding, cand_indexes):
        # word_embedding is defined by nn.Embedding and has shape [vocab_size, hidden_size]
        batch_size, max_seq_len, max_chunk_per_word = cand_indexes.size()

        cand_indexes_2d = cand_indexes.view(batch_size*max_seq_len*max_chunk_per_word, 1)

        # word_embedding has shape [batch_size*max_seq_len*max_chunk_per_word, hidden_size]
        cand_embedding_2d = word_embedding(cand_indexes_2d)

        # [batch_size, max_seq_len, max_chunk_per_word, hidden_size]
        cand_embedding_3d = cand_embedding_2d.view(batch_size, max_seq_len, max_chunk_per_word, -1)

        # [batch_size, max_seq_len, hidden_size]
        embedding_output = torch.mean(cand_embedding_3d, dim=2, keepdim=False)

        embedding_output = torch.sum(
            cand_embedding_3d*cand_mask, dim=2, keepdim=False) / torch.sum(cand_mask, dim=2)
        return embedding_output

    def update_embedding(self, word_embedding, cand_indexes):
        new_word_embedding = torch.zeros_like(word_embedding)
        word_embedding_i = torch.zeros_like(word_embedding[0])


        for i, cand_index_s in enumerate(cand_indexes): # each sentence
            # copy feature for [CLS]
            word_embedding_i[0] = word_embedding[i][0].clone()
            last_sel_index = 0

            for j, cand_index in enumerate(cand_index_s): # cand_index for each sentence
                if cand_index[0] == -1: # end of sentence
                    break

                tmp_mask = cand_index.ge(0)
                sel_index = torch.masked_select(cand_index, tmp_mask)
                last_sel_index = sel_index[-1]

                if self.update_method == 'mean':
                    feat = torch.mean(word_embedding[i][sel_index], dim=0, keepdim=True)

                word_embedding_i[j+1] = feat.clone()

            # copy feature for [SEP]
            word_embedding_i[j+1] = word_embedding[i][last_sel_index+1].clone()

            new_word_embedding[i] = word_embedding_i.clone()

        return new_word_embedding


class BertMLModel(PreTrainedBertModel):
    """Modified from BERT models ("Bidirectional Embedding Representations from a Transformer").
        for multilinguisticss

    Params:
        config: a BertConfig class instance with the configuration to build a new models

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
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    cand_indexes = torch.LongTensor([[0, 1, 2], [0, 1]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    models = modeling.BertMLModel(config=config)
    all_encoder_layers, pooled_output = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, update_method='mean'):
        super(BertMLModel, self).__init__(config)
        #self.update_method = update_method
        self.embeddings = BertMLEmbeddings(config, update_method)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, \
                cand_indexes=None, token_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, attention_mask, cand_indexes, token_ids)

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertMLVariantCWSPOS(PreTrainedBertModel):
    """Apply BERT for Sequence Labeling on Chinese Word Segmentation and Part-of-Speech with multilinguistics.

    models = BertMIVariantCWSPOS(config, num_tags)
    logits = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_CWStags=6, num_POStags=108, method='fine_tune', fclassifier='Softmax', \
                 pclassifier='Softmax', do_mask_as_whole=False):
        super(BertMLVariantCWSPOS, self).__init__(config)
        self.num_CWStags = num_CWStags
        self.num_POStags = num_POStags
        self.method = method
        self.fclassifier = fclassifier  # cws classifier
        self.pclassifier = pclassifier  # pos classifier
        self.do_mask_as_whole = do_mask_as_whole

        if self.do_mask_as_whole:
            self.bert = BertMLModel(config)
        else:
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
        self.hidden2CWStag = nn.Linear(last_hidden_size, num_CWStags)
        self.hidden2POStag = nn.Linear(last_hidden_size, num_POStags)

        if self.fclassifier == 'CRF':
            self.CWSclassifier = CRF(num_CWStags, batch_first=True)

        if self.pclassifier == 'CRF':
            self.POSclassifier = CRF(num_POStags, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cand_indexes=None, token_ids=None,
                labels_CWS=None, labels_POS=None):
        mask = attention_mask.byte()
        loss = 1e10

        if labels_CWS is None and labels_POS is None:
            raise RuntimeError('Input: labels_CWS or labels_POS is missing!')
        else:
            feat_used = self._compute_bert_feats(input_ids, token_type_ids, attention_mask, cand_indexes, token_ids)

            if labels_CWS is not None:
                cws_logits = self.hidden2CWStag(feat_used)
                loss = self._compute_loss(cws_logits, mask, labels_CWS, 'CWS')

            if labels_POS is not None:
                pos_logits = self.hidden2POStag(feat_used)
                loss += self._compute_loss(pos_logits, mask, labels_POS, 'POS')

        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, cand_indexes=None, token_ids=None, \
               labels_CWS=None, labels_POS=None):
        cws_loss = 1e10
        pos_loss = 1e10

        mask = attention_mask.byte()
        feat_used = self._compute_bert_feats(input_ids, token_type_ids, attention_mask, cand_indexes, token_ids)

        cws_logits = self.hidden2CWStag(feat_used)
        pos_logits = self.hidden2POStag(feat_used)

        if labels_CWS is not None:
            cws_loss = self._compute_loss(cws_logits, mask, labels_CWS, 'CWS')

        if labels_POS is not None:
            pos_loss = self._compute_loss(pos_logits, mask, labels_POS, 'POS')

        if self.fclassifier == 'CRF':
            best_cws_tags_list = self.CWSclassifier.decode(cws_logits, mask)
        elif self.fclassifier == 'Softmax':
            best_cws_tags_list = self._decode_Softmax(cws_logits, mask)

        if self.pclassifier == 'CRF':
            best_pos_tags_list = self.POSclassifier.decode(pos_logits, mask)
        elif self.pclassifier == 'Softmax':
            best_pos_tags_list = self._decode_Softmax(pos_logits, mask)

        return cws_loss, pos_loss, best_cws_tags_list, best_pos_tags_list

    def _compute_bert_feats(self, input_ids, token_type_ids=None, attention_mask=None, cand_indexes=None, token_ids=None):
        if self.method in ['last_layer', 'fine_tune']:
            output_all_encoded_layers = False
        else: # sum_last4, sum_all, cat_last4, 'MHMLA'
            output_all_encoded_layers = True

        if self.do_mask_as_whole:
            if cand_indexes is None and token_ids is None:
                raise RuntimeError('Input: cand_indexes and token_ids are missing!')

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, \
                   output_all_encoded_layers=output_all_encoded_layers, \
                   cand_indexes=cand_indexes, token_ids=token_ids)

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

        return feat_used

    def _compute_loss(self, logits, mask, labels, task):
        # mask is a ByteTensor

        if task == 'CWS':
            if self.fclassifier == 'Softmax':
                loss_fct = nn.CrossEntropyLoss()

                num_tags = self.num_CWStags
                loss = loss_fct(logits.view(-1, num_tags), labels.view(-1))
            elif self.fclassifier == 'CRF':
                loss = -self.CWSclassifier(logits, labels, mask)
        elif task == 'POS':
            if self.pclassifier == 'Softmax':
                loss_fct = nn.CrossEntropyLoss()
                num_tags = self.num_POStags

                loss = loss_fct(logits.view(-1, num_tags), labels.view(-1))
            elif self.pclassifier == 'CRF':
                loss = -self.POSclassifier(logits, labels, mask)

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


class BertMLCWSPOS(BertMLVariantCWSPOS):
    """Apply BERT for Sequence Labeling on Chinese Word Segmentation and Part-of-Speech.

    models = BertMLCWSPOS(device, config, vocab_file, max_length, num_CWStags=6, num_POStags=110, batch_size=64, fclassifier='Softmax', method='fine_tune')
    logits = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, device, config, vocab_file, max_length, num_CWStags=6, num_POStags=110, batch_size=64,
                 do_lower_case=False, do_mask_as_whole=False, fclassifier='Softmax', pclassifier='Softmax', \
                 method='fine_tune'):
        super(BertMLCWSPOS, self).__init__(config)
        BertMLVariantCWSPOS.__init__(self, config, num_CWStags=num_CWStags, num_POStags=num_POStags, method=method,
                 fclassifier=fclassifier, pclassifier=pclassifier, do_mask_as_whole=do_mask_as_whole)
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_length = max_length

    def _seg_wordslist(self, lword):  # ->str
        # lword: list of words (list)
        # input_ids, segment_ids, input_mask = tokenize_list(
        #     words, self.max_length, self.tokenizer)
        #print(lword)
        tuple1, tuple2, tuple3 = zip(
            *[tokenize_list_with_cand_indexes_lang_status(w, self.max_length, self.tokenizer) for w in lword if w]) # w is not empty
            #*[tokenize_list(w, self.max_length, self.tokenizer) for w in lword])
            #*[tokenize_list_no_seg(w, self.max_length, self.tokenizer) for w in lword])
        list1 = unpackTuple(tuple1)
        input_ids = list1[0::3]
        segment_ids = list1[1::3]
        input_masks = list1[2::3]

        list2 = unpackTuple(tuple2)
        cand_indexes = list2[0::2]
        token_ids = list2[1::2]

        lang_status = unpackTuple(tuple3)
        #lang_status = list3[0::]

        input_id_torch = torch.from_numpy(np.array(input_ids)).to(self.device)
        segment_ids_torch = torch.from_numpy(np.array(segment_ids)).to(self.device)
        input_masks_torch = torch.from_numpy(np.array(input_masks)).to(self.device)
        cand_indexes_troch = torch.from_numpy(np.array(cand_indexes)).to(self.device)
        token_ids_torch = torch.from_numpy(np.array(token_ids)).to(self.device)
        lang_status_torch = torch.from_numpy(np.array(lang_status)).to(self.device)

        _, _, best_cws_tags_list, best_pos_tags_list = self.decode(input_id_torch, segment_ids_torch, \
                                           input_masks_torch, cand_indexes_troch, token_ids_torch)

        cws_output_list = []
        for idx, rs in enumerate(best_cws_tags_list):
            cws_decode_output = ''.join(str(v) for v in rs[1:-1]) #

            # tmp_rs[1:-1]: remove the tokens, [START] and [END]
            #decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
            # i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}
            # i.e., BMES_idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'M', 4: 'E', 5: 'S'}

            # replace the [START] and [END] tokens
            # predict those wrong tokens as a separated word
            # replacing 0 and 1 should not be conducted usually
            cws_decode_output = cws_decode_output.replace(str(segType.BMES_label_map['[START]']), str(segType.BMES_label_map['S']))
            cws_decode_output = cws_decode_output.replace(str(segType.BMES_label_map['[END]']), str(segType.BMES_label_map['S']))

            if 1:
                lang_status_i = lang_status_torch[idx]
                cws_decode_output_l = list(cws_decode_output)
                for ii, ls_ii in enumerate(lang_status_i):
                    if ls_ii==1: cws_decode_output_l[ii] = str(segType.BMES_label_map['S'])
                cws_decode_output = ''.join(cws_decode_output_l)

            cws_output_list.append(cws_decode_output)

        pos_output_list = []
        for rs in best_pos_tags_list:
            pos_decode_output = ' '.join(posType.POS_label_map[(v-2)//3] if v > 2 else posType.POS_label_map[35] for v in rs[1:-1]) #

            # tmp_rs[1:-1]: remove the tokens, [START] and [END]
            #decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens in POSType.BIO_idx_to_label_map

            # replace the [START] and [END] tokens ??
            # predict those wrong tokens as a separated word
            # replacing 0 and 1 should not be conducted usually
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[START]']), str(segType.BMES_label_map['S']))
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[END]']), str(segType.BMES_label_map['S']))

            pos_output_list.append(pos_decode_output)

        return cws_output_list, pos_output_list  # list of string

    def cutlist_noUNK(self, input_list):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            models = BertCWS(config, num_tags, vocab_file, max_length)
            output = models.cutlist_noUNK([text])
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

        original_text_list = processed_text_list
        processed_text_list = [self.tokenizer.tokenize(t) if len(self.tokenizer.tokenize(t))>0 \
                               else ['[UNK]'] for t in processed_text_list]

        cws_output_list = []
        pos_output_list = []
        tmp_pos_list = []

        batch_size = self.batch_size
        for p_t_l in [processed_text_list[0+i:batch_size+i] for i in range(0, len(processed_text_list), batch_size)]:
            #print(p_t_l)
            #if '翡翠' in ''.join(p_t_l[0]):
            #    print('test')
            #if len(p_t_l)==0:
            #    cws_output = segType.BMES_label_map['S']
            #    pos_output = posType.POS2idx_map['PU']
            #    continue # avoid input empty tokens

            cws_output, pos_output = self._seg_wordslist(p_t_l)
            cws_output_list.extend(cws_output)
            pos_output_list.extend(pos_output)

        # restoring processed_text_list to list of strings
        #processed_text_list = [''.join(char_list) for char_list in processed_text_list]
        result_str_list = []

        for merge_start, merge_end in merge_index_list:
            result_str = ''
            original_str = ''
            result_pos = '' # storing pos results

            cws_tag = ''.join(cws_output_list[merge_start:merge_end])
            pos_tag = ' '.join(pos_output_list[merge_start:merge_end]).split()

            text = []
            for a in processed_text_list[merge_start:merge_end]:
                cand_indexes = define_words_set(a)

                for idx_ls in cand_indexes:
                    pa = ''
                    for idx in idx_ls:
                        pa += a[idx].replace('##', '')
                    text.append(pa)

            for a in original_text_list[merge_start:merge_end]:
                str_used = ''
                al = re.split('[\n\r]', a)

                for aa in al: str_used += ''.join(aa.strip())

                original_str += str_used

            tmp_pos = []
            seg_start = False
            for idx in range(len(cws_tag)):
                tt = text[idx]
                tt = tt.replace('##', '')
                ti = cws_tag[idx]
                pos_tag_i = pos_tag[idx]

                try:
                    int(ti)
                except ValueError:
                    print(ti + '\n')      # or whatever
                    print(cws_tag)

                int_ti = int(ti)
                if int_ti == segType.BMES_label_map['B']:  # 'B'
                    result_str += ' ' + tt

                    if not seg_start:
                        seg_start = True

                    if tmp_pos != []:
                        tmp_pos_list.append(tmp_pos)

                    result_pos += pos_tag_i + ' '
                    tmp_pos = [pos_tag_i]
                elif int_ti > segType.BMES_label_map['M']:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '

                    #if int_ti == segType.BMES_label_map['S']:
                    #    result_pos += pos_tag_i + ' '
                    #    tmp_pos = []

                    #    tmp_pos_list.append([pos_tag_i])
                    #else:
                    if tmp_pos == []:
                        result_pos += pos_tag_i + ' '

                    tmp_pos.extend([pos_tag_i])
                    tmp_pos_list.append(tmp_pos)
                    tmp_pos = []

                    seg_start = False
                else:
                    result_str += tt
                    tmp_pos.append(pos_tag_i)
                    if not seg_start:
                        seg_start = True
                        result_pos += pos_tag_i + ' '

            result_pos_str = extract_pos(tmp_pos_list)

            if '[UNK]' in result_str or '[unused' in result_str:
                print(original_str)
                seg_ls, pos_ls = restore_unknown_tokens_without_unused_with_pos(original_str, result_str, result_pos)
            else:
                seg_ls = result_str.strip().split()
                pos_ls = result_pos.strip().split()

            #seg_ls = result_str_rev.strip().split()
            #pos_ls = result_pos_rev.strip().split()
            assert(len(seg_ls)==len(pos_ls))

            rs = []
            for i in range(len(seg_ls)):
                rs.append(seg_ls[i] + ' / ' + pos_ls[i])

            result_str_list.append(rs)

        return result_str_list


class BertMLVariantCWSPOS_with_Dict(BertMLVariantCWSPOS):
    """Apply BERT for Sequence Labeling on Chinese Word Segmentation and Part-of-Speech with multilinguistics
    and input features processed from dictionary.

    The class is inherited from BertMLVariantCWSPOS, where the functions, _compute_loss and _decode_Softmax, are
    inherited from BertMLVariantCWSPOS.

    models = BertMLVariantCWSPOS_with_Dict(config, num_CWStags, ...)
    logits = models(input_ids, token_type_ids, input_mask, ...)
    ```
    """
    def __init__(self, config, num_CWStags=6, num_POStags=108, method='fine_tune', fclassifier='Softmax', \
                 pclassifier='Softmax', do_mask_as_whole=False, dict_file=None):
        super(BertMLVariantCWSPOS_with_Dict, self).__init__(config)
        self.num_CWStags = num_CWStags
        self.num_POStags = num_POStags
        self.method = method
        self.fclassifier = fclassifier  # cws classifier
        self.pclassifier = pclassifier  # pos classifier
        self.do_mask_as_whole = do_mask_as_whole

        if dict_file is not None:
            self.dict = read_dict(dict_file)
            self.max_gram = MAX_GRAM_LEN # default is 16
            #config.hidden_size += (self.max_gram-1)*2 # if no dictionary, self.max_gram=1
        else:
            self.max_gram = 1 # if no dictionary

        if self.do_mask_as_whole:
            self.bert = BertMLModel(config)
        else:
            self.bert = BertModel(config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.last_hidden_size = self._set_last_hidden_size(method)

        # Maps the output of BERT into tag space.
        self.hidden2CWStag = nn.Linear(self.last_hidden_size, num_CWStags)
        self.hidden2POStag = nn.Linear(self.last_hidden_size, num_POStags)

        if self.fclassifier == 'CRF':
            self.CWSclassifier = CRF(num_CWStags, batch_first=True)

        if self.pclassifier == 'CRF':
            self.POSclassifier = CRF(num_POStags, batch_first=True)

        self.apply(self.init_bert_weights)

    def _set_last_hidden_size(self, method='fine_tune'):
        if method == 'fine_tune':
            last_hidden_size = self.config.hidden_size + (self.max_gram-1)*2
        elif method == 'cat_last4':
            self.biLSTM = nn.LSTM(input_size=self.config.hidden_size*4,
                                  hidden_size=self.config.hidden_size,
                                  num_layers=2, batch_first=True,
                                  dropout=0, bidirectional=True)
            last_hidden_size = self.config.hidden_size*2 + (self.max_gram-1)*2
        elif method in ['last_layer', 'sum_last4', 'sum_all', 'cat_last4']:
            self.biLSTM = nn.LSTM(input_size=self.config.hidden_size,
                                  hidden_size=self.config.hidden_size,
                                  num_layers=2, batch_first=True,
                                  dropout=0, bidirectional=True)
            last_hidden_size = self.config.hidden_size*2 + (self.max_gram-1)*2
        elif self.method == 'MHMLA':
            self.MHMLA = MultiHeadMultiLayerAttention(config)
            last_hidden_size = self.config.hidden_size + (self.max_gram-1)*2
        return last_hidden_size

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cand_indexes=None, token_ids=None, \
                input_via_dict=None, labels_CWS=None, labels_POS=None):
        mask = attention_mask.byte()
        loss = 1e10

        if labels_CWS is None and labels_POS is None:
            raise RuntimeError('Input: labels_CWS or labels_POS is missing!')
        else:
            feat_used = self._compute_bert_feats(input_ids, token_type_ids, attention_mask, cand_indexes, token_ids, input_via_dict)

            if labels_CWS is not None:
                cws_logits = self.hidden2CWStag(feat_used)
                loss = self._compute_loss(cws_logits, mask, labels_CWS, 'CWS')

            if labels_POS is not None:
                pos_logits = self.hidden2POStag(feat_used)
                loss += self._compute_loss(pos_logits, mask, labels_POS, 'POS')

        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None, cand_indexes=None, token_ids=None,
               input_via_dict=None, labels_CWS=None, labels_POS=None):
        cws_loss = 1e10
        pos_loss = 1e10

        mask = attention_mask.byte()
        feat_used = self._compute_bert_feats(input_ids, token_type_ids, attention_mask, cand_indexes, token_ids, input_via_dict)

        cws_logits = self.hidden2CWStag(feat_used)
        pos_logits = self.hidden2POStag(feat_used)

        if labels_CWS is not None:
            cws_loss = self._compute_loss(cws_logits, mask, labels_CWS, 'CWS')

        if labels_POS is not None:
            pos_loss = self._compute_loss(pos_logits, mask, labels_POS, 'POS')

        if self.fclassifier == 'CRF':
            best_cws_tags_list = self.CWSclassifier.decode(cws_logits, mask)
        elif self.fclassifier == 'Softmax':
            best_cws_tags_list = self._decode_Softmax(cws_logits, mask)

        if self.pclassifier == 'CRF':
            best_pos_tags_list = self.POSclassifier.decode(pos_logits, mask)
        elif self.pclassifier == 'Softmax':
            best_pos_tags_list = self._decode_Softmax(pos_logits, mask)

        return cws_loss, pos_loss, best_cws_tags_list, best_pos_tags_list

    def _compute_bert_feats(self, input_ids, token_type_ids=None, attention_mask=None, cand_indexes=None, token_ids=None, \
                            input_via_dict=None):
        if self.method in ['last_layer', 'fine_tune']:
            output_all_encoded_layers = False
        else: # sum_last4, sum_all, cat_last4, 'MHMLA'
            output_all_encoded_layers = True

        if self.do_mask_as_whole:
            if cand_indexes is None and token_ids is None:
                raise RuntimeError('Input: cand_indexes and token_ids are missing!')

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, cand_indexes=cand_indexes,
                               token_ids=token_ids, output_all_encoded_layers=output_all_encoded_layers)

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

        if input_via_dict is not None:
            input_via_dict = input_via_dict.to(dtype=next(self.parameters()).dtype)
            feat_used = torch.cat((feat_used, input_via_dict), 2)

        return feat_used


class BertMLCWSPOS_with_Dict(BertMLVariantCWSPOS_with_Dict):
    """Apply BERT for Sequence Labeling on Chinese Word Segmentation and Part-of-Speech.

    models = BertMLCWSPOS_with_Dict(device, config, vocab_file, max_length, num_CWStags=6, num_POStags=110, batch_size=64, fclassifier='Softmax', method='fine_tune')
    logits = models(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, device, config, vocab_file, max_length, num_CWStags=6, num_POStags=110, batch_size=64,
                 do_lower_case=False, do_mask_as_whole=False, fclassifier='Softmax', pclassifier='Softmax', \
                 method='fine_tune', dict_file=None):
        super(BertMLCWSPOS_with_Dict, self).__init__(config)
        BertMLVariantCWSPOS_with_Dict.__init__(self, config, num_CWStags=num_CWStags, num_POStags=num_POStags, \
                    method=method, fclassifier=fclassifier, pclassifier=pclassifier, \
                    do_mask_as_whole=do_mask_as_whole,
                    dict_file=dict_file)

        self.device = device
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_length = max_length

        #if dict_file is not None:
        #    self.dict_mat = torch.zeros((max_length, (self.max_gram-1)*2), device=device)

    def _seg_wordslist(self, lword):  # ->str
        # lword: list of words (list)
        # input_ids, segment_ids, input_mask = tokenize_list(
        #     words, self.max_length, self.tokenizer)
        #print(lword)
        tuple1, tuple2, tuple3, input_via_dict = zip(
            *[tokenize_list_with_cand_indexes_lang_status_dict_vec(w, self.max_length, self.tokenizer, self.dict)
              for w in lword if w]) # , self.dict_mat
            #*[tokenize_list_with_cand_indexes_lang_status(w, self.max_length, self.tokenizer) for w in lword if w]) # w is not empty
            #*[tokenize_list(w, self.max_length, self.tokenizer) for w in lword])
            #*[tokenize_list_no_seg(w, self.max_length, self.tokenizer) for w in lword])
        list1 = unpackTuple(tuple1)
        input_ids = list1[0::3]
        segment_ids = list1[1::3]
        input_masks = list1[2::3]

        list2 = unpackTuple(tuple2)
        cand_indexes = list2[0::2]
        token_ids = list2[1::2]

        lang_status = unpackTuple(tuple3)
        input_via_dict = unpackTuple(input_via_dict)
        #lang_status = list3[0::]

        input_id_torch = torch.from_numpy(np.array(input_ids)).to(self.device)
        segment_ids_torch = torch.from_numpy(np.array(segment_ids)).to(self.device)
        input_masks_torch = torch.from_numpy(np.array(input_masks)).to(self.device)
        cand_indexes_troch = torch.from_numpy(np.array(cand_indexes)).to(self.device)
        token_ids_torch = torch.from_numpy(np.array(token_ids)).to(self.device)
        lang_status_torch = torch.from_numpy(np.array(lang_status)).to(self.device)
        input_via_dict_torch = torch.from_numpy(np.array(input_via_dict)).to(self.device)

        _, _, best_cws_tags_list, best_pos_tags_list = self.decode(input_id_torch, segment_ids_torch, \
                                           input_masks_torch, cand_indexes_troch, token_ids_torch, input_via_dict_torch)

        cws_output_list = []
        for idx, rs in enumerate(best_cws_tags_list):
            cws_decode_output = ''.join(str(v) for v in rs[1:-1]) #

            # tmp_rs[1:-1]: remove the tokens, [START] and [END]
            #decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens corresponding to B, M, E, S, [START], [END],
            # i.e, BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}
            # i.e., BMES_idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'M', 4: 'E', 5: 'S'}

            # replace the [START] and [END] tokens
            # predict those wrong tokens as a separated word
            # replacing 0 and 1 should not be conducted usually
            cws_decode_output = cws_decode_output.replace(str(segType.BMES_label_map['[START]']), str(segType.BMES_label_map['S']))
            cws_decode_output = cws_decode_output.replace(str(segType.BMES_label_map['[END]']), str(segType.BMES_label_map['S']))

            if 1:
                lang_status_i = lang_status_torch[idx]
                cws_decode_output_l = list(cws_decode_output)
                for ii, ls_ii in enumerate(lang_status_i):
                    if ls_ii==1: cws_decode_output_l[ii] = str(segType.BMES_label_map['S'])
                cws_decode_output = ''.join(cws_decode_output_l)

            cws_output_list.append(cws_decode_output)

        pos_output_list = []
        for rs in best_pos_tags_list:
            pos_decode_output = ' '.join(posType.POS_label_map[(v-2)//3] if v > 2 else posType.POS_label_map[35] for v in rs[1:-1]) #

            # tmp_rs[1:-1]: remove the tokens, [START] and [END]
            #decode_output = tmp_rs[1:-1]

            # Now decode_output should consists of the tokens in POSType.BIO_idx_to_label_map

            # replace the [START] and [END] tokens ??
            # predict those wrong tokens as a separated word
            # replacing 0 and 1 should not be conducted usually
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[START]']), str(segType.BMES_label_map['S']))
            #decode_output = decode_output.replace(str(segType.BMES_label_map['[END]']), str(segType.BMES_label_map['S']))

            pos_output_list.append(pos_decode_output)

        return cws_output_list, pos_output_list  # list of string

    def cutlist_noUNK(self, input_list):
        """
        # Example usage:
            text = '''
            目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
            ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
            （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
            # of students is 256.
            '''

            models = BertCWS(config, num_tags, vocab_file, max_length)
            output = models.cutlist_noUNK([text])
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

        original_text_list = processed_text_list
        processed_text_list = [self.tokenizer.tokenize(t) if len(self.tokenizer.tokenize(t))>0 \
                               else ['[UNK]'] for t in processed_text_list]

        cws_output_list = []
        pos_output_list = []
        tmp_pos_list = []

        batch_size = self.batch_size
        for p_t_l in [processed_text_list[0+i:batch_size+i] for i in range(0, len(processed_text_list), batch_size)]:
            #print(p_t_l)
            #if '翡翠' in ''.join(p_t_l[0]):
            #    print('test')
            #if len(p_t_l)==0:
            #    cws_output = segType.BMES_label_map['S']
            #    pos_output = posType.POS2idx_map['PU']
            #    continue # avoid input empty tokens

            cws_output, pos_output = self._seg_wordslist(p_t_l)
            cws_output_list.extend(cws_output)
            pos_output_list.extend(pos_output)

        # restoring processed_text_list to list of strings
        #processed_text_list = [''.join(char_list) for char_list in processed_text_list]
        result_str_list = []

        for merge_start, merge_end in merge_index_list:
            result_str = ''
            original_str = ''
            result_pos = '' # storing pos results

            cws_tag = ''.join(cws_output_list[merge_start:merge_end])
            pos_tag = ' '.join(pos_output_list[merge_start:merge_end]).split()

            text = []
            for a in processed_text_list[merge_start:merge_end]:
                cand_indexes = define_words_set(a)

                for idx_ls in cand_indexes:
                    pa = ''
                    for idx in idx_ls:
                        pa += a[idx].replace('##', '')
                    text.append(pa)

            for a in original_text_list[merge_start:merge_end]:
                str_used = ''
                al = re.split('[\n\r]', a)

                for aa in al: str_used += ''.join(aa.strip())

                original_str += str_used

            tmp_pos = []
            seg_start = False
            for idx in range(len(cws_tag)):
                tt = text[idx]
                tt = tt.replace('##', '')
                ti = cws_tag[idx]
                pos_tag_i = pos_tag[idx]

                try:
                    int(ti)
                except ValueError:
                    print(ti + '\n')      # or whatever
                    print(cws_tag)

                int_ti = int(ti)
                if int_ti == segType.BMES_label_map['B']:  # 'B'
                    result_str += ' ' + tt

                    if not seg_start:
                        seg_start = True

                    if tmp_pos != []:
                        tmp_pos_list.append(tmp_pos)

                    result_pos += pos_tag_i + ' '
                    tmp_pos = [pos_tag_i]
                elif int_ti > segType.BMES_label_map['M']:  # and (cur_word_is_english)
                    # int(ti)>1: tokens of 'E' and 'S'
                    # current word is english
                    result_str += tt + ' '

                    #if int_ti == segType.BMES_label_map['S']:
                    #    result_pos += pos_tag_i + ' '
                    #    tmp_pos = []

                    #    tmp_pos_list.append([pos_tag_i])
                    #else:
                    if tmp_pos == []:
                        result_pos += pos_tag_i + ' '

                    tmp_pos.extend([pos_tag_i])
                    tmp_pos_list.append(tmp_pos)
                    tmp_pos = []

                    seg_start = False
                else:
                    result_str += tt
                    tmp_pos.append(pos_tag_i)
                    if not seg_start:
                        seg_start = True
                        result_pos += pos_tag_i + ' '

            result_pos_str = extract_pos(tmp_pos_list)

            if '[UNK]' in result_str or '[unused' in result_str:
                print(original_str)
                seg_ls, pos_ls = restore_unknown_tokens_with_pos(original_str, result_str, result_pos)
            else:
                seg_ls = result_str.strip().split()
                pos_ls = result_pos.strip().split()

            #seg_ls = result_str_rev.strip().split()
            #pos_ls = result_pos_rev.strip().split()
            assert(len(seg_ls)==len(pos_ls))

            rs = []
            for i in range(len(seg_ls)):
                rs.append(seg_ls[i] + ' / ' + pos_ls[i])

            result_str_list.append(rs)

        return result_str_list

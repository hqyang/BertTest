import torch
import torch.nn as nn
import math
from src.BERT.modeling import PreTrainedBertModel, BertModel
from src.TorchCRF import CRF

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


'''
BertForSequenceLabeling
class BertForSequenceLabeling(PreTrainedBertModel):
    """BERT model for Sequence Labeling.
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

    model = BertForSequenceLabeling(config, num_tags)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_tags=3):
        super(BertForSequenceLabeling, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = CRF(num_tags)
        #nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
'''


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

    model = Bert_CRF(config, num_tags)
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
            loss = self.classifier(bert_feats, labels, mask)
            decode_result = self.classifier.decode(bert_feats, mask)
            return loss, decode_result


class BertCRF3Layer(PreTrainedBertModel):
    """BERT model with CRF for Sequence Labeling.
    This module is composed of the BERT model with a linear layer on top of the third layer
    output via Conditional Random Field.

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

    model = Bert_CRF(config, num_tags)
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
            loss = self.classifier(bert_feats, labels, mask)
            decode_result = self.classifier.decode(bert_feats, mask)
            return loss, decode_result





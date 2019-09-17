from pprint import pprint

class Config:
    ## Required parameters
    task_name = None  # MSR， PKU, AS, CITYU, ONTONOTES
    data_dir = None
    bert_config_file = None
    vocab_file = None
    output_dir = None
    bert_model_dir = None

    ## Other parameters
    ##1.Basic tasks
    do_train = False
    do_eval = False
    do_eval_df = False
    trainBERT = True
    bert_model = None

    ##2.Preprocessing
    do_lower_case = False
    nopunc = False
    do_mask_as_whole = True
    dict_file = './resource/dict.txt'

    ##3.Training configs
    init_checkpoint = None
    seed = 42
    max_seq_length = 128
    train_batch_size = 128
    eval_batch_size = 128
    learning_rate = 2e-5
    num_train_epochs = 3.0
    warmup_proportion = 0.1
    num_hidden_layers = 0
    projected_size = 6
    method = 'fine_tune' # 'last_layer', 'cat_last4', 'sum_last4', 'sum_all'
    fclassifier = 'Softmax' # 'CRF'
    pclassifier = 'CRF'

    ##4.Devices
    no_cuda = False
    local_rank = -1
    gradient_accumulation_steps = 1
    optimize_on_cpu = False
    fp16 = False
    loss_scale = 128
    visible_device = None
    
    #5.Task options
    isResume = False
    modelIdx = 0
    override_output = False
    multilabel = False
    multitask = False
    pretraining = False
    modification = None
    model_type = None

    def _parse(self, kwargs, verbose=True):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        if verbose:
            print('======user config========')
            pprint(self._state_dict())
            print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

args = Config()

class SegType:
    BMES_idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'M', 4: 'E', 5: 'S'}
    BIO_idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'I', 4: 'O'}
    BMES_label_map = {'[START]': 0, '[END]': 1, 'B': 2, 'M': 3, 'E': 4, 'S': 5}

segType = SegType()

class POSType:
    BIO_idx_to_label_map = {
        0: '[START]', 1: '[END]', 2: 'B-AD', 3: 'I-AD', 4: 'O-AD',
        5: 'B-AS', 6: 'I-AS', 7: 'O-AS', 8: 'B-BA', 9: 'I-BA', 10: 'O-BA',
        11: 'B-CC', 12: 'I-CC', 13: 'O-CC', 14: 'B-CD', 15: 'I-CD', 16: 'O-CD',
        17: 'B-CS', 18: 'I-CS', 19: 'O-CS', 20: 'B-DEC', 21: 'I-DEC', 22: 'O-DEC',
        23: 'B-DEG', 24: 'I-DEG', 25: 'O-DEG', 26: 'B-DER', 27: 'I-DER', 28: 'O-DER',
        29: 'B-DEV', 30: 'I-DEV', 31: 'O-DEV', 32: 'B-DT', 33: 'I-DT', 34: 'O-DT',
        35: 'B-ETC', 36: 'I-ETC', 37: 'O-ETC', 38: 'B-FW', 39: 'I-FW', 40: 'O-FW',
        41: 'B-IJ', 42: 'I-IJ', 43: 'O-IJ', 44: 'B-INF', 45: 'I-INF', 46: 'O-INF',
        47: 'B-JJ', 48: 'I-JJ', 49: 'O-JJ', 50: 'B-LB', 51: 'I-LB', 52: 'O-LB',
        53: 'B-LC', 54: 'I-LC', 55: 'O-LC', 56: 'B-M', 57: 'I-M', 58: 'O-M',
        59: 'B-MSP', 60: 'I-MSP', 61: 'O-MSP', 62: 'B-NN', 63: 'I-NN', 64: 'O-NN',
        65: 'B-NR', 66: 'I-NR', 67: 'O-NR', 68: 'B-NT', 69: 'I-NT', 70: 'O-NT',
        71: 'B-OD', 72: 'I-OD', 73: 'O-OD', 74: 'B-ON', 75: 'I-ON', 76: 'O-ON',
        77: 'B-P', 78: 'I-P', 79: 'O-P', 80: 'B-PN', 81: 'I-PN', 82: 'O-PN',
        83: 'B-PU', 84: 'I-PU', 85: 'O-PU', 86: 'B-SB', 87: 'I-SB', 88: 'O-SB',
        89: 'B-SP', 90: 'I-SP', 91: 'O-SP', 92: 'B-URL', 93: 'I-URL', 94: 'O-URL',
        95: 'B-VA', 96: 'I-VA', 97: 'O-VA', 98: 'B-VC', 99: 'I-VC', 100: 'O-VC',
        101: 'B-VE', 102: 'I-VE', 103: 'O-VE', 104: 'B-VV', 105: 'I-VV', 106: 'O-VV',
        107: 'B-X', 108: 'I-X', 109: 'O-X'
    }
    POS_label_map = {
        0: 'AD', 1: 'AS', 2: 'BA', 3: 'CC', 4: 'CD', 5: 'CS', 6: 'DEC', 7: 'DEG',
        8: 'DER', 9: 'DEV', 10: 'DT', 11: 'ETC', 12: 'FW', 13: 'IJ', 14: 'INF',
        15: 'JJ', 16: 'LB', 17: 'LC', 18: 'M', 19: 'MSP', 20: 'NN', 21: 'NR',
        22: 'NT', 23: 'OD', 24: 'ON', 25: 'P', 26: 'PN', 27: 'PU', 28: 'SB',
        29: 'SP', 30: 'URL', 31: 'VA', 32: 'VC', 33: 'VE', 34: 'VV', 35: 'X'
    }

    POS2idx_map = {
        'AD': 0, 'AS': 1, 'BA': 2, 'CC': 3, 'CD': 4, 'CS': 5, 'DEC': 6, 'DEG': 7,
        'DER': 8, 'DEV': 9, 'DT': 10, 'ETC': 11, 'FW': 12, 'IJ': 13, 'INF': 14,
        'JJ': 15, 'LB': 16, 'LC': 17, 'M': 18, 'MSP': 19, 'NN': 20, 'NR': 21,
        'NT': 22, 'OD': 23, 'ON': 24, 'P': 25, 'PN': 26, 'PU': 27, 'SB': 28,
        'SP': 29, 'URL': 30, 'VA': 31, 'VC': 32, 'VE': 33, 'VV': 34, 'X': 35
    }

posType = POSType()

    # explain from https://blog.csdn.net/Eliza1130/article/details/40678999
    # 0 AD: 副词  Adverbs
    # 1 AS    语态词  --- 了
    # 2 BA    把
    # 3 CC    并列连接词（coordinating conj）
    # 4 CD    许多(many),若干（several),个把(a,few)
    # 5 CS    从属连接词（subording conj）
    # 6 DEC   从句“的”
    # 7 DEG   修饰“的”
    # 8 DER   得 in V-de-const, and V-de R
    # 9 DEV   地 before VP
    # 10 DT    限定词   各（each),全(all),某(certain/some),这(this)
    # 11 ETC   for words 等，等等
    # 12 FW    外来词 foreign words
    # 13 IJ     感叹词  interjecton
    # 14 INF     那个
    # 15 JJ     名词修饰语
    # 16 LB    被,给   in long bei-const
    # 17 LC    方位词
    # 18 M     量词
    # 19 MSP   其他小品词（other particle） 所
    # 20 NN    口头名词、others
    # 21 NR    专有名词
    # 22 NT    时间名词  （temporal noun）
    # 23 OD    序数（ordinal numbers）
    # 24 ON    拟声法（onomatopoeia）
    # 25 P      介词   （对，由于，因为）(除了 “把”和“被”)
    # 26 PN    代词
    # 27 PU    标定符号
    # 28 SB    in short bei-const 被，给
    # 29 SP    句尾语气词
    # 30 URL
    # 31 VA    表语形容词（predicative adjective）
    # 32 VC    是
    # 33 VE    有（have，not have ,有，无，没，表示存在的词
    # 34 VV    情态动词、  动词、possess/拥有 ，rich/富有,具有
    # 35 X     English x

UNK_TOKEN = "[UNK]"
PUNC_TOKENS = "(＂|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|\u3000|、|〃|〈|〉|《|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏|﹑|﹔|·|！|？|｡|。|')"

MAX_SUBWORDS = 64 # train: 10; test: 64

UNUSED_SPACE_TOKEN = '[UnUsed_!@#]'

# suppose the last length is 16 for two eight-words idioms
# however, to make the hidden size is not a multiple of the number of attention heads (12),
# we set it to 25
MAX_GRAM_LEN = 25

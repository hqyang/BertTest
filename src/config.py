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
    append_dir = False  # evaluate the performance of training data, work when do_eval = True
    trainBERT = True    
    bert_model = None

    ##2.Preprocessing
    do_lower_case = True
    nopunc = False
    
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
posType = POSType()

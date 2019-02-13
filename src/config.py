from pprint import pprint

class Config:
    
    ## Required parameters
    task_name = None
    data_dir = None
    bert_config_file = None
    vocab_file = None
    output_dir = None
    bert_model_dir = None
    
    ## Other parameters
    ##1.Basic tasks
    do_train = False
    do_eval = False
    
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
    
    ##4.Devices
    no_cuda = False
    local_rank = -1
    gradient_accumulation_steps = 1
    optimize_on_cpu = False
    fp16 = False
    loss_scale = 128
    visible_device = None
    
    #5.Task options
    override_output = False
    multilabel = False
    multitask = False
    pretraining = False
    modification = None
    model_type = None
    tensorboardWriter = False

    #6.Pre-trained options
    retrained_model_dir = None

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
    BMES_idx_to_label_map = {0: 'B', 1: 'M', 2: 'E', 3: 'S', 4: '[START]', 5: '[END]'}
    BIO_idx_to_label_map = {0: 'B', 1: 'I', 2: 'O', 3: '[START]', 4: '[END]'}

segType = SegType()

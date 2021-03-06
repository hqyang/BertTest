import re

from ..tokenization import FullTokenizer

from ..utils import (
    get_or_make_label_encoder,
    create_single_problem_generator,
    BOS_TOKEN,
    EOS_TOKEN)


def parse_one(s):
    s = re.sub('\)', ') ', s)
    s = re.sub(' +', ' ', s).strip()
    pos = s.split(' ')
    buffer = []
    innermost = True
    full_pos = []
    seg = []
    ner = []
    ner_types = []
    text = []
    for p in pos:
        if '(' in p:
            innermost = True
            if 'NER' in p:
                ner_types.append(re.sub('NER', '', p[1:]))
                continue
            else:
                buffer.append(p)
        elif ')' in p:
            if buffer != []:
                suffix = buffer.pop()[1:]
                if innermost:
                    assert len(p) > 1
                    word = p[:-1]
                    text.append(word)
                    innermost = False
                    p = p[-1]
                    if len(word) == 1:
                        seg_gt = ['O']
                    else:
                        seg_gt = ['B'] + ['M'] * (len(word) - 2) + ['E']
                    if ner_types != []:
                        ner_type = ner_types.pop()
                        ner_gt = [_ + ner_type for _ in seg_gt]
                    else:
                        ner_gt = ['O' for _ in seg_gt]
                    seg.extend(seg_gt)
                    ner.extend(ner_gt)
            p += suffix
        full_pos.append(p)
    text = list(''.join(text))
    return seg, ner, full_pos, text


def ontonotes_ner(params, mode):
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)

    if mode == 'train':
        with open('data/ontonote/train.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()
    else:
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()

    _, target, _, inputs_list = zip(*[parse_one(s) for s in raw_data])
    flat_target_list = [t for sublist in target for t in sublist]
    label_encoder = get_or_make_label_encoder(
        params, 'ontonotes_ner', mode, flat_target_list)

    return create_single_problem_generator('ontonotes_ner',
                                           inputs_list,
                                           target,
                                           label_encoder,
                                           params,
                                           tokenizer)


def ontonotes_cws(params, mode):
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)

    if mode == 'train':
        with open('data/ontonote/train.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()
    else:
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()

    target, _, _, inputs_list = zip(*[parse_one(s) for s in raw_data])
    flat_target_list = [t for sublist in target for t in sublist]
    label_encoder = get_or_make_label_encoder(
        params, 'ontonotes_cws', mode, flat_target_list)

    return create_single_problem_generator('ontonotes_cws',
                                           inputs_list,
                                           target,
                                           label_encoder,
                                           params,
                                           tokenizer)


def ontonotes_chunk(params, mode):

    tokenizer = FullTokenizer(vocab_file=params.vocab_file)

    if mode == 'train':
        with open('data/ontonote/train.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()
    else:
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()

    _, _, target, inputs_list = zip(*[parse_one(s) for s in raw_data])
    flat_target_list = [t for sublist in target for t in sublist]
    flat_target_list.extend([BOS_TOKEN, EOS_TOKEN])
    label_encoder = get_or_make_label_encoder(
        params, 'ontonotes_chunk', mode, flat_target_list)
    params.eos_id['ontonotes_chunk'] = label_encoder.transform([EOS_TOKEN])[0]

    return create_single_problem_generator(
        'ontonotes_chunk',
        inputs_list,
        target,
        label_encoder,
        params,
        tokenizer)

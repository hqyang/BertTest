## History
# Test classifier
python run_classifier_torch.py

# Test Chinese Word Segmentation
python main.py

# Input format for the Ontonotes dataset
*   bert_ner	bert_seg	full_pos	src_ner	src_seg	text	text_seg
>  An example
>   #   s = '(NP (CP (IP (NP (DNP (NER-GPE (NR Taiwan)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))'
>   #   bert_ner: W-GPE O B-ORG E-ORG B E B E O B E B E B M M E B M E O
>   #   bert_seg: S S B E B E B E S B E B E B M M E B M E S
>   #   full_pos: (NP (CP (IP (NP (DNP (NR )NR )DNP (DEG )DEG )NP (NR )NR )IP )CP (VP (NT )NT (VV )VV )VP )NP (DEC )DEC )DEC (NP-m (NP (NR )NR (NN )NN )NP (NP-m (NP (NN )NN (NN )NN )NP (PU )PU )NP-m )NP-m )NP-m
>   #   src_ner: W-GPE,O,B-ORG,E-ORG,B,E,B,E,O,B,E,B,E,O,O,O,
>   #   src_seg: S,S,B,E,B,E,B,E,S,B,E,B,E,S,S,S,
>   #   text: Taiwan的公视今天主办的台北市长candidate defence，
>   #   text_seg: Taiwan 的 公视 今天 主办 的 台北 市长 candidate defence ，

# Input format for the four CWS datasets: AS, CityU, MSR, PKU
*   bert_seg	full_pos	src_seg	text	text_seg
>  An example
>   #   s = '目前　由　２３２　位　院士　（　Ｆｅｌｌｏｗ　及　Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ　）　，６６　位　協院士　（　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ　）　２４　位　通信　院士　（　Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ　）　及　２　位　通信　協院士　（　Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ　）　組成　（　不　包括　一九九四年　當選　者　）　，'
>   #   bert_seg: S S B E B E B E S B E B E B M M E B M E S
>   #   src_seg: S,S,B,E,B,E,B,E,S,B,E,B,E,S,S,S,
>   #   text: 目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者），
>   #   text_seg: 目前　由　２３２　位　院士　（　Ｆｅｌｌｏｗ　及　Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ　）　，６６　位　協院士　（　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ　）　２４　位　通信　院士　（　Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ　）　及　２　位　通信　協院士　（　Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ　）　組成　（　不　包括　一九九四年　當選　者　）　，

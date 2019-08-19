from tokenizations import tokenization_bert_word_level as tokenization_bert
import ujson as json
import thulac
from tqdm import tqdm
lac = thulac.thulac(seg_only=True)

def generate_vocab(data_path, output_path):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = json.load(f)
    s = set()
    def process(line):
        words = lac.cut(line)
        words = [word[0] for word in words]
        ts = set(words)
        s.update(ts)
    #import pdb;pdb.set_trace()
    _ = list(map(process, lines))
    keys = [word for word in s]
    with open(output_path, 'w') as fh:
        keys = [key + '\n' for key in ['[SEP]', '[PAD]','[CLS]', '[MASK]', '[UNK]'] + keys]
        fh.writelines(keys)
#generate_vocab('data/train.json', 'cache/vocab_wiki_small_new.txt')

full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab_wiki_small.txt')
def tokenize_list(word_list):
    for word in word_list:
        print(full_tokenizer.convert_tokens_to_ids(word))
tokenize_list(['中国', '政府', '今天', '猫'])
#with open('/home/t-linan/projects/GPT2-Chinese/data/tokenized/tokenized_train_0.txt', 'r') as fh:
with open('data/tokenized/tokenized_train_0_False.txt', 'r') as fh:
    line = fh.readlines()[0]
numbers = line.strip().split()
print(len(numbers))
print(sum([int(number)== 4 for number in numbers]))
"""
with open('../chineseLM/output_wiki_ch_small_cwt_39697_reverseFalse/vocab.json', 'r') as fh:
    counter = json.load(fh)['counter']
keys = list(counter.keys())
with open('vocab.txt', 'w') as fh:
    keys = [key + '\n' for key in ['[SEP]', '[PAD]','[CLS]', '[MASK]', '[UNK]'] + keys]
    fh.writelines(keys)
"""
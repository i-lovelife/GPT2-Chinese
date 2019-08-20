import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
from tokenizations import tokenization_bert_word_level
import argparse
import numpy as np
from tqdm import trange
from pytorch_transformers import GPT2LMHeadModel
import itertools
from operator import itemgetter
from generate import sample_sequence

class Gpt2Vocab:
    UNKNOWN_CHAR = '[UNK]'
    MASK_CHAR = '[MASK]'
    END_CHAR = '[SEP]'
    START_CHAR = '[CLS]'
    PAD_CHAR = '[PAD]'
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.tokenizer._convert_id_to_token(idx)
        elif isinstance(idx, str):
            return self.tokenizer._convert_token_to_id(idx)
        raise ValueError(f'not recognized {idx}')
    def sentence2idx(self, sentence, require_tokenize=True):
        if require_tokenize:
            sentence = self.tokenizer.tokenize(sentence)
        return self.tokenizer.convert_tokens_to_ids(sentence)

class Gpt2LanguageModel:
    """
    #  Prepare tokenized input

        text_1 = "Who was Jim Henson ?"

        text_2 = "Jim Henson was a puppeteer"

        indexed_tokens_1 = tokenizer.encode(text_1)

        indexed_tokens_2 = tokenizer.encode(text_2)

        tokens_tensor_1 = torch.tensor([indexed_tokens_1])

        tokens_tensor_2 = torch.tensor([indexed_tokens_2])
        with torch.no_grad():

            predictions_1, past = model(tokens_tensor_1)

            predictions_2, past = model(tokens_tensor_2, past=past)

            # Get the predicted last token

            predicted_index = torch.argmax(predictions_2[0, -1, :]).item()

            predicted_token = tokenizer.decode([predicted_index])

            assert predicted_token == ' who'
    """
    def __init__(self, model_path, tokenizer_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = tokenization_bert_word_level.BertTokenizer(vocab_file=tokenizer_path)
        vocab = Gpt2Vocab(tokenizer)
        self.device = device
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer

    def get_init_state(self):
        return None

    def inference_from_state(self, new_char, old_state=None):
        self.model.eval()
        X = torch.tensor([[self.vocab[new_char]]]).to(self.device)
        with torch.no_grad():
            logits, new_state = self.model(X, past=old_state)[:2]
            Y = -torch.log(F.softmax(logits, dim=-1))
            Y = Y[0, -1, :].cpu().numpy()
        return Y, new_state

    def get_sentence_prob(self, sentence, require_tokenize=True):
        # Y_pred: (batch_size, seq_len, vocab_size)
        # Y: (batch_size, seq_len)
        self.model.eval()
        vocab = self.vocab
        indexed_sentence = vocab.sentence2idx(sentence, require_tokenize=require_tokenize)
        X = torch.tensor([[vocab[vocab.START_CHAR]] + indexed_sentence + [vocab[vocab.END_CHAR]]]).to(self.device)
        Y_target = X
        hidden = None
        with torch.no_grad():
            loss, Y_pred, hidden = self.model(X, labels=Y_target, past=hidden)[:3]
        return loss.item()

    def generate_sentence(self):
        context = self.tokenizer.convert_tokens_to_ids(self.vocab.START_CHAR)
        for i in range(10):
            out = sample_sequence(self.model, num_samples=1, length=30, context=context, device=self.device)
            text = self.tokenizer.convert_ids_to_tokens(out[0].tolist())
            print(' '.join(text))
        #print(text)

def test_lm():
    model_path = 'model/final_model/'
    tokenizer_path = 'cache/vocab_wiki_small.txt'
    sentence = '有唐一代，山西一直以其特殊的地位和发达的经济、文化称著于世'
    forward_lm = Gpt2LanguageModel(model_path, tokenizer_path)

    prob = 0
    forward_prob = forward_lm.get_sentence_prob(sentence)
    old_state = None
    indexed_sentence = forward_lm.vocab.sentence2idx(sentence, require_tokenize=True)
    tokenized_sentence = forward_lm.tokenizer.tokenize(sentence)
    prob_history = []
    for i in range(len(tokenized_sentence) + 1):
        new_char = tokenized_sentence[i-1] if i > 0 else forward_lm.vocab.START_CHAR
        y_pred, old_state = forward_lm.inference_from_state(new_char=new_char, old_state=old_state)
        if i < len(tokenized_sentence):
            prob_history.append(y_pred[forward_lm.vocab[tokenized_sentence[i]]])
        else:
            prob_history.append(y_pred[forward_lm.vocab[forward_lm.vocab.END_CHAR]])
    prob = sum(prob_history) / len(prob_history)
    print(forward_prob, prob)
    assert np.isclose(forward_prob, prob, 1e-3)
#test_lm()

def predict(model,
            constraint_list,
            max_sentence_length=10,
            beam_size=10,
            soft_threshold=None, 
            similarity_calculator=None,
            **args):
    # turn constraint into index
    constraint_list = [model.vocab.sentence2idx(item) for item in constraint_list]
    # calculate whether require extend for each num of constraint has satisfied
    require_extend = [[True] * (len(item) - 1) + [False] for item in constraint_list]
    constraint_list = list(itertools.chain.from_iterable(constraint_list))
    require_extend = list(itertools.chain.from_iterable(require_extend))
    # init candidate groups with [[]*(len(constraint_list)+1)]
    candidate_groups = [[([model.vocab.START_CHAR], model.get_init_state(), 0., [])]] + [[] for i in range(len(constraint_list))]
    # init best score and best candidate
    best_score = 1e9
    best_sentence = None
    best_score_history = []

    if soft_threshold is not None:
        constraint_char = [model.vocab[idx] for idx in constraint_list]
        similarity_matrix = similarity_calculator.calculate_similarity_list(constraint_char, model.vocab.idx2char)

    forbidden_prefix = ('TAG', 
                        ',', 
                        '\ue40c', 
                        '.', 
                        model.vocab.UNKNOWN_CHAR,
                        model.vocab.MASK_CHAR,
                        model.vocab.END_CHAR,
                        model.vocab.START_CHAR)

    for step in range(max_sentence_length + 1):
        next_candidate_groups = [[] for i in range(len(constraint_list) + 1)]
        for satisfied_num, candidates in enumerate(candidate_groups):
            candidates = sorted(candidates, key=itemgetter(2))
            if len(candidates) > beam_size:
                candidates = candidates[:beam_size]
            for sentence, state, score, score_history in candidates:
                new_char = sentence[-1]
                # calculate nll score and new state
                nll_scores, new_state = model.inference_from_state(old_state=state, new_char=new_char)
                def create_new_sentence(idx, group_id=satisfied_num, weight=1.):
                    new_sentence = sentence + [model.vocab[idx]]
                    new_score = score + nll_scores[idx] * weight
                    new_score_history = score_history + [nll_scores[idx] * weight]
                    next_candidate_groups[group_id].append((new_sentence, new_state, new_score, new_score_history))
                    return True
                #t1 = time.time()
                if satisfied_num == 0 or require_extend[satisfied_num - 1] is False:
                    # not fulfill any, choose best beam_size candidate
                    next_char_indexes = np.argsort(nll_scores)
                    next_char_indexes = list(filter(lambda id: model.vocab[id].startswith(forbidden_prefix) is False,
                                                    next_char_indexes))
                    if len(next_char_indexes) > beam_size:
                        next_char_indexes = next_char_indexes[:beam_size]
                    _ = list(map(create_new_sentence, next_char_indexes))
                if satisfied_num < len(constraint_list):
                    # fulfill one constraint
                    if soft_threshold is None:# hard
                        next_char_indexes = [constraint_list[satisfied_num]]
                    else:# soft 
                        next_char_indexes = [idx for idx, score in enumerate(similarity_matrix[satisfied_num, :]) if score > soft_threshold]
                    next_char_indexes = list(filter(lambda id: model.vocab[id].startswith(forbidden_prefix) is False,
                                                    next_char_indexes))
                    if len(next_char_indexes) > beam_size:
                        next_char_indexes = next_char_indexes[:beam_size]
                    _ = list(map(lambda x:create_new_sentence(x, satisfied_num + 1, 10.), next_char_indexes))
                # update answer
                end_char = model.vocab.END_CHAR
                end_idx = model.vocab[end_char]
                end_score = score + nll_scores[end_idx]
                end_score_history = score_history + [nll_scores[end_idx]]
                if satisfied_num == len(constraint_list) and end_score / len(sentence) < best_score:
                    best_score = end_score / len(sentence)
                    best_sentence = sentence
                    best_score_history = end_score_history
                #print(f'elapsed fulfill{time.time()-t1} s')    
        candidate_groups = next_candidate_groups
    for word, score in zip(best_sentence[1:], best_score_history):
        print(f'{word} {score}')
    return best_sentence[1:], best_score_history
        
def test_grid_beam_search():
    model_path = 'model/backwardmodel_epoch9/'
    tokenizer_path = 'cache/vocab_wiki_small.txt'
    sentence = '有唐一代，山西一直以其特殊的地位和发达的经济、文化称著于世'
    forward_lm = Gpt2LanguageModel(model_path, tokenizer_path)
    forward_lm.generate_sentence()
    
    #predict(forward_lm, constraint_list=['山西', '经济'])
if __name__ == '__main__':
    test_grid_beam_search()

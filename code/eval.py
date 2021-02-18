import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
#from utils import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval
import numpy as np
import heapq

# Parameters
data_folder = '../prepared_data'  # folder with data files saved by create_input_files.py
#data_name = 'flickr8kzh_5_cap_per_img_5_min_word_freq_seg_based'  # base name shared by data files
#checkpoint = '../checkpoints/BEST_checkpoint_' + data_name + '_fine_tune.pth.tar'  # model checkpoint
#word_map_file = '../prepared_data/WORDMAP_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
#print("device = ", device)
#cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
'''
# Load model
checkpoint = torch.load(checkpoint, 
                        #map_location=lambda storage, loc: storage.cuda(1)
                       )
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)
word_map_start = word_map['<start>']
word_map_end = word_map['<end>']'''

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate_with_beam(beam_width, data_name, model, encoder, decoder, word_map, word_map_start, word_map_end, rev_word_map):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    references_zh = list()
    hypotheses_zh = list()
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_width))):
        #if i % 1000 != 0: continue 
        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Decode
        seq, _ = decode_one(decoder, encoder_out, encoder_dim, enc_image_size, word_map_start, word_map_end, beam_width)

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        #img_captions_zh = [" ".join(rev_word_map[word] for word in sentence) for sentence in img_captions]
        img_captions_zh = [" ".join(str(word) for word in sentence) for sentence in img_captions]

        references.append(img_captions)
        references_zh.append(img_captions_zh)
        

        # Hypotheses
        hypothese = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(hypothese)
        #hypothese_zh = [rev_word_map[item] for item in hypothese]
        hypothese_zh = [str(item) for item in hypothese]
        hypothese_zh = " ".join(hypothese_zh)
        hypotheses_zh.append(hypothese_zh)
        
        assert len(references) == len(hypotheses)
        assert len(references_zh) == len(hypotheses_zh)

        # Calculate BLEU-4 scores
    bleu1nltk = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2nltk = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3nltk = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4nltk = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    nlgeval = NLGEval()  # loads the models
    metrics_dict = nlgeval.compute_metrics(references_zh, hypotheses_zh)
    metrics_dict["bleu1nltk"]=bleu1nltk
    metrics_dict["bleu2nltk"]=bleu2nltk
    metrics_dict["bleu3nltk"]=bleu3nltk
    metrics_dict["bleu4nltk"]=bleu4nltk
    
    # write result to json file
    #print(metrics_dict)
    output = {"model": model[15:-8], "beam_width" : beam_width, "scores" : metrics_dict}
    
    output_file_name = "../evaluation/"+ model[15:-8] + ".txt"
    f = open(output_file_name,"a+")
    f.writelines(json.dumps(str(output)))
    f.writelines("\n")
    f.close()
    return metrics_dict

class beam():
    def __init__(self, beam_width=5):
        self.heap = list()
        self.beam_width = beam_width
        
    def add(self, prob, complete, seq, alphas, inputs, h, c):
        heapq.heappush(self.heap, [prob, complete, seq, alphas, inputs, h, c])
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)
    
    def __iter__(self):
        return iter(self.heap)
    
def decode_one(decoder, encoder_out, encoder_dim, enc_image_size, word_map_start, word_map_end, beam_width):
    """Generate one sample"""
    batch_size = 1
    inputs = torch.Tensor([word_map_start]).long().to(device)
    h, c = decoder.init_hidden_state(encoder_out)
    alphas = torch.ones(1, enc_image_size, enc_image_size).to(device)
    prev_beam = beam(beam_width)
    prev_beam.add(1, False, [word_map_start], alphas, inputs, h, c)
    while True:
        cur_beam = beam()
        for _prob, _complete, _seq, _alphas, _inputs, _h, _c in prev_beam:
            if _complete == True:
                cur_beam.add(_prob, _complete, _seq, _alphas, _inputs, _h, _c)
            else:
                embeddings = decoder.embedding(_inputs)  # (1, embed_dim)
                awe, alpha = decoder.attention(encoder_out, _h)  # (1, encoder_dim), (1, num_pixels)
                alpha = alpha.view(-1, enc_image_size, enc_image_size) # (1, enc_image_size, enc_image_size)
                gate = decoder.sigmoid(decoder.f_beta(_h))  # gating scalar, (1, encoder_dim)
                awe = gate * awe
                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (_h, _c))
                score = decoder.fc(h)  # (1, vocab_size)
                preds = F.softmax(score, dim=1)
                value, pred = torch.topk(preds.view(-1),beam_width)
                for m, n in zip(value, pred):
                    next_input = n.item()
                    inputs = torch.Tensor([next_input]).long().to(device)
                    seq = _seq + [n.item()]
                    prob = _prob * m.item()
                    alphas = torch.cat((_alphas, alpha), dim=0)
                    if n.item() == word_map_end or len(seq) == 51:
                        complete = True
                    else:
                        complete = False
                    cur_beam.add(prob, complete, seq, alphas, inputs, h, c)
        best_prob, best_complete, best_seq, best_alphas, _, _, _ = max(cur_beam)
        if best_complete == True:
            #del embeddings, awe, alpha, gate, h, c, score, preds, value, pred, next_input, inputs, seq, prob, alphas
            return best_seq, best_alphas
        else:                
            prev_beam = cur_beam
'''if __name__ == '__main__':
    for beam_width in range(1,6):
        print(data_name)
        print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_width, evaluate_with_beam(beam_width)))'''

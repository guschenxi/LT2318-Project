import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import heapq
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='../library/heiti.ttf', size=12)
import torch.nn.functional as F
import matplotlib.cm as cm
import skimage.transform
import argparse
from imageio import imread
from PIL import Image
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder, max_len=100, char_based=False):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'flickr8kzh', 'flickr30kzh', 'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            if char_based == False:
                tokens = c['tokens'].split()
                word_freq.update(tokens)
                if len(tokens) <= max_len:captions.append(c['tokens'])
            elif char_based == True:
                tokens = "".join(c['tokens'].split())
                word_freq.update(tokens)
                if len(tokens) <= max_len:captions.append(c['tokens'])
            #captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    if char_based == True:
        a = "_char_based"
    elif char_based == False:
        a = "_seg_based"
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'+ a

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    if char_based == True:
                        # Encode captions
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c if word!=" "] + [
                            word_map['<end>']] 
                        # Find caption lengths
                        c_len = len("".join(c.split())) + 2
                    elif char_based == False:
                        # Encode captions
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c.split()] + [
                            word_map['<end>']]
                        # Find caption lengths
                        c_len = len(c.split()) + 2
                    # pad
                    enc_c = enc_c + [word_map['<pad>']] * (max_len + 2 - len(enc_c))
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)

def read_image_and_resize(image_path):
    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    #img = imresize(img, (256, 256))
    img = np.array(Image.fromarray(img).resize((256,256), Image.BICUBIC))
    
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    return image

def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, fine_tune):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    path = "../checkpoints/"
    if fine_tune:
        filename = 'checkpoint_' + data_name + '_fine_tune.pth.tar'
    else:
        filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, 
               path + filename, 
               _use_new_zipfile_serialization=False
              )
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state,
                   path + 'BEST_' + filename,
                   _use_new_zipfile_serialization=False
                  )


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

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
            
def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    fig = plt.figure(figsize=(20, 5))
    for t in range(len(words)):
        if t > 50:
            break
        
        plt.subplot(np.ceil(len(words) / 10.), 10, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12, fontproperties=font)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.detach().numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.detach().numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
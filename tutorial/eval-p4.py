import torch
import json
from eval import evaluate_with_beam
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

data_folder = '../prepared_data'  # folder with data files saved by create_input_files.py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
print("torch.version=", torch.__version__)
print("device=",device)
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

from checkpoints4 import data_names, models, word_maps

for data_name, model, word_map_file in zip(data_names, models, word_maps):

    print("Model: ", model)
    # Load model
    checkpoint = torch.load(model, 
                            map_location=lambda storage, loc: storage.cuda(0)
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
    word_map_end = word_map['<end>']
    #print(vocab_size, word_map_start)
    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Evaluate with beam search
    for beam_width in range(1,6):
        score = evaluate_with_beam(beam_width, data_name, model, encoder, decoder, word_map, word_map_start, word_map_end, rev_word_map)
        print(score)
        print()

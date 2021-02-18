from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8kzh',
                       karpathy_json_path='../data/flickr8kzh.json',
                       image_folder='../data/flickr8k_images/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../prepared_data/',
                       max_len=50,
                       char_based=False)

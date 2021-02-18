data_name1 = "flickr30kzh_5_cap_per_img_5_min_word_freq_seg_based"
data_name2 = "flickr30kzh_5_cap_per_img_5_min_word_freq_char_based"
data_name3 = "flickr8kzh_5_cap_per_img_5_min_word_freq_seg_based"
data_name4 = "flickr8kzh_5_cap_per_img_5_min_word_freq_char_based"

VGG19, ft = ['',''] 
model1 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name1 + ft + ".pth.tar"
model2 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name2 + ft + ".pth.tar"
model3 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name3 + ft + ".pth.tar"
model4 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name4 + ft + ".pth.tar"
model1nb = "../checkpoints/checkpoint_" + VGG19 + data_name1 + ft + ".pth.tar"
model2nb = "../checkpoints/checkpoint_" + VGG19 + data_name2 + ft + ".pth.tar"
model3nb = "../checkpoints/checkpoint_" + VGG19 + data_name3 + ft + ".pth.tar"
model4nb = "../checkpoints/checkpoint_" + VGG19 + data_name4 + ft + ".pth.tar"

VGG19, ft = ['','_fine_tune'] 
model5 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name1 + ft + ".pth.tar"
model6 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name2 + ft + ".pth.tar"
model7 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name3 + ft + ".pth.tar"
model8 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name4 + ft + ".pth.tar"
model5nb = "../checkpoints/checkpoint_" + VGG19 + data_name1 + ft + ".pth.tar"
model6nb = "../checkpoints/checkpoint_" + VGG19 + data_name2 + ft + ".pth.tar"
model7nb = "../checkpoints/checkpoint_" + VGG19 + data_name3 + ft + ".pth.tar"
model8nb = "../checkpoints/checkpoint_" + VGG19 + data_name4 + ft + ".pth.tar"

VGG19, ft = ['VGG19_',''] 
model9 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name1 + ft + ".pth.tar"
model10 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name2 + ft + ".pth.tar"
model11 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name3 + ft + ".pth.tar"
model12 = "../checkpoints/BEST_checkpoint_" + VGG19 + data_name4 + ft + ".pth.tar"
model9nb = "../checkpoints/checkpoint_" + VGG19 + data_name1 + ft + ".pth.tar"
model10nb = "../checkpoints/checkpoint_" + VGG19 + data_name2 + ft + ".pth.tar"
model11nb = "../checkpoints/checkpoint_" + VGG19 + data_name3 + ft + ".pth.tar"
model12nb = "../checkpoints/checkpoint_" + VGG19 + data_name4 + ft + ".pth.tar"

word_map1 ="../prepared_data/WORDMAP_" + data_name1 + ".json"
word_map2 ="../prepared_data/WORDMAP_" + data_name2 + ".json"
word_map3 ="../prepared_data/WORDMAP_" + data_name3 + ".json"
word_map4 ="../prepared_data/WORDMAP_" + data_name4 + ".json"

data_names = [
              #data_name1, halv
              data_name2, #data_name3, data_name4,
              ]

models = [
          #model5, halv
          model6, #model7, model8, 
          ]

word_maps = [
             #word_map1, 
             word_map2, #word_map3, word_map4,
             ]
        
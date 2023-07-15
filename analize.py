import numpy as np
from pre_processing import pre_processing
import argparse, os
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

parser = argparse.ArgumentParser()
parser.add_argument('--dir-name', type=str,  help='name of directory for saving model')
args = parser.parse_args()

model_dir = os.path.join("model",args.dir_name)

theta = np.load(os.path.join(model_dir,"theta.npy"))
phi = np.load(os.path.join(model_dir,"phi.npy"))
z = np.load(os.path.join(model_dir,"z.npy"))
index2word_dict = np.load(os.path.join(model_dir,"index2word_dict.npy"),allow_pickle=True).item()
text_k = []
K = 10
print(phi.shape)
for k in range(K):
    # word_list = np.random.choice(phi.shape[1],size=(100,),p=phi[k])
    word_list = np.argsort(phi[k])[::-1][:200]
    text= ""
    for word in word_list:
        text = text +" "+ index2word_dict[word]
    text_k.append(text)

#可視化
font_path = '/content/ShipporiMincho-Regular.ttf'
# fig, axs = plt.subplots(ncols=K, nrows=1, figsize=(12,12))
# axs = axs.flatten()

def color_func(word, font_size, position, orientation, random_state, font_path):
    return 'darkturquoise'

#ワードクラウドを楕円形にするためのイメージをmaskとして取得します。
# mask = np.array(Image.open("/content/phpYSbfIJ.png"))

for k in range(K):
    wordcloud = WordCloud(max_font_size=40,max_words=50).generate(text_k[k])
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.imshow(wordcloud)
    plt.savefig(model_dir+f"/{k}.svg")
#     axs[k].imshow(wordcloud)
#     axs[k].axis('off')
#     axs[k].set_title('Topic '+str(k))

# plt.tight_layout()
# plt.show()
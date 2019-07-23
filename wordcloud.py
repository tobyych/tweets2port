import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

datapath = "../data/stocknet-dataset/tweet/preprocessed"
folderpath = [x[0] for x in os.walk(datapath)][1:]
print(folderpath)
for folder in folderpath:
    print(folder)
    filelist = os.listdir(folder)
    filelist.sort()
    concat_string = ""
    for filename in filelist:
        with open(os.path.join(folder, filename), "r", encoding="ascii") as f:
            line = f.readline()
            text = ast.literal_eval(line)["text"]  # convert string to dict
            concat_string += (
                " ".join(text).encode("utf-16", "surrogatepass").decode("utf-16") + " "
            )
    concat_string = " ".join(
        [x for x in concat_string.split() if x not in set(filler_tokens)]
    )
    print(concat_string)
    wc = WordCloud(font_path="/Library/Fonts/AppleGothic.ttf")
    wc.generate(concat_string)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    break

import pandas as pd
from pathlib import Path
from collections import Counter
from nltk.tokenize import casual_tokenize
PATH = 'dialogues/'


# tried to install nlpia, failed repeatedly, went w/ this instead of get_data
def reader(file_path):
    contents = Path(file_path).read_text()
    return contents


# user input prompt
def prompt():
    user_input = input('please enter your question: ')
    return user_input


# processes data strings
def process(raw):
    for r in range(len(raw)):
        raw[r] = ' '+raw[r].split('"turns": ')[1][1:-3].replace('",', ' ')
        raw[r] = raw[r].replace(' "', '')+'\n'
    return raw


# retrieves dialogue through feature name
def retrieve(title):
    file, number = title.rsplit('_', 1)
    # print(file+'.txt', number)
    prefix = (file+'.txt, '+number)
    dialogue = prefix+': '+process(reader(PATH+file+'.txt').splitlines())[int(number)]
    return dialogue


# since attempting to read & create a bow for all 47 files at once would likely result in a memory crash
# i decided to construct 47 bows one .txt file at a time, pick the 10 best-fitting dialogues from each
# & then proceed to compare dialogue lists w/ every iteration
# keeping only the 10 overall best results in the end

# gets names from dir.txt
names = reader('dir.txt').splitlines()  # instead of entering the .txt file names by hand
names = [name.rsplit(' ', 1)[1] for name in names]  # i decided to use a dir printout for ease
print(names)  # debugging print
new_sentence = prompt().lower()  # compatible with our normalisation for the dialogue data
final = []
total = 0
for name in names:
    # string processing
    data = process(reader(PATH+name).splitlines())

    m = len(str((len(data))))  # used to format dialogue names
    bags_of_words = []
    for d in data:
        bags_of_words.append(Counter(casual_tokenize(d.lower())))  # decided to lower for normalisation

    # formatting each feature by filename & dialogue number
    features = ['{}_{}{}'.format(name[:-4], (m-len(str(i)))*'0', i) for i in range(len(bags_of_words))]
    # print(features)  # debugging print

    # appends new sentence to bow & feature list
    bags_of_words.append(Counter(casual_tokenize(new_sentence)))
    features.append('input')

    # creates data frame
    df_bows = pd.DataFrame.from_records(bags_of_words, features)
    df_bows = df_bows.fillna(0).astype(int)
    print(df_bows.head())
    total += df_bows.shape[0]-1
    df_bows = df_bows.T
    print(df_bows.head())

    # dictionary of products
    dp = {i: df_bows.input.dot(df_bows.loc[:, i]) for i in features[:-1]}

    # temporary product list
    temp = sorted(dp.items(), key=lambda x: x[1], reverse=True)[:10]
    print(temp)  # debugging print
    final = sorted((final+temp), key=lambda x: x[1], reverse=True)[:10]

print(final)  # final list of the 10 closest results (highest dot products)
print('total dialogues: ', total)
results = [retrieve(f[0]) for f in final]
results = ''.join(results)
print('here are the 10 closest dialogues:\n{}'.format(results))

from tqdm import tqdm
from io import open
import csv

# Seq2seq reads files where source and target are written on one line with a tab
# \t in between

def extract_conversations(filename='data/dialog/dialogs.txt'):
    print('Get separate conversations...')
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    conversations = []
    tmp_conv = []
    for line in tqdm(lines):
        if len(line) > 2:  # "\r\n" - newline
            tmp_conv.append(line[:-1].split(' '))
        else:
            conversations.append(tmp_conv)
            tmp_conv = []
    print('Number of conversations: ', len(conversations))
    return conversations


def write_pairs(convs, output_filename='seq2seq_dialogs.tsv'):
    print("\nWriting dialog pairs to {}.")
    with open(output_filename, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=' ')
        for conv in tqdm(convs):
            for i in range(len(conv)-1):
                writer.writerow(conv[i] + ['\t'] + conv[i+1])


if __name__ == "__main__":
    #  Write txt file
    convs = extract_conversations()
    write_pairs(convs)

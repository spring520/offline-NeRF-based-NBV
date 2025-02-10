import sys
sys.path.append("/home/zhengquan/04-fep-nbv")
from fep_nbv.utils import offset2word


if __name__=='__main__':
    # 输入：偏移量和词性
    offset = '02958343'  # 示例偏移量
    word = offset2word(offset)
    print(word)

    '''pos = 'n'      # 词性：'n' 表示名词, 'v' 表示动词, 'a' 形容词, 'r' 副词

    # 使用 synset_from_pos_and_offset 获取 Synset
    synset = wn.synset_from_pos_and_offset(pos, offset)

    # 获取同义词集中所有单词
    words = synset.lemma_names()

    # 输出结果
    print("Synset:", synset)
    print("Words in the synset:", words)'''

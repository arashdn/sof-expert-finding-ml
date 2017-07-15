one_gram_file_path = "./data/grams/1gram.csv"

def get_distinct_words(vocab_size=65536):  ## get word and return two indexes, first word->term frquncy then rank frequency -> word
    res = dict()
    countword = 0
    with open(one_gram_file_path) as f:
        lines = f.readlines()
        for l in lines:
            sp = l.split(",")
            countword += 1
            if sp[1] in res:
                res[sp[1]] += 1
            else:
                res[sp[1]] = 1
    tmp = sorted(res.items(), key=lambda x: x[1], reverse=True)
    tmp = tmp[:min(countword, vocab_size)]
    out = ''

    for t, _ in tmp:
        out += t
        out += '\n'

    ff = open("./data/top_words.txt", "w")
    print(out, file=ff)
    ff.close()
    
    # final_res = dict()
    # for t, _ in tmp:
    #     final_res[t] = len(final_res)
    # final_res2 = {v: k for k, v in final_res.items()}
    # return final_res, final_res2

get_distinct_words()
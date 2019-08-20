def ConvertBinaryBracketedSeq(seq):
    T_SHIFT = 0
    T_REDUCE = 1
    T_SKIP = 2

    tokens, transitions = [], []
    for item in seq:
        if item != "(":
            if item != ")":
                tokens.append(item)
            transitions.append(T_REDUCE if item == ")" else T_SHIFT)
    return tokens, transitions
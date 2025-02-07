from curses.ascii import isdigit

# These functions translate between the FEN and (homegrown) sequence (seq) format
# The seq format works as follows:
# # It is a string of len 64
# # Each index corresponding to a specific field, 0 being top left, 63 being bottom right
# # All pieces use the same key as the FEN format, except an empty field is a '_' (there is no row divider)


def FEN_to_seq(FEN):
    sequence = ""
    assert isinstance(FEN, str)
    for i in range(len(FEN)):
        if FEN[i] == '-': continue
        if isdigit(FEN[i]) : sequence = sequence + '_' * int(FEN[i])
        else: sequence = sequence + FEN[i]
    assert len(sequence) == 64
    return sequence

def seq_to_FEN(seq):
    assert isinstance(seq, str)
    FEN = ""
    blankspaceit = 0

    def add_blankspaces_contingently():
        nonlocal FEN, blankspaceit
        if blankspaceit > 0:
            FEN = FEN + str(blankspaceit)
            blankspaceit = 0
    
    row = 0        
    while row < 8 :
        column = 0
        while column < 8 :
            
            currentItem = seq[(8 * row) + column]
            if currentItem == '_':
                blankspaceit += 1
            else:
                add_blankspaces_contingently()
                FEN = FEN + currentItem
            column += 1
        add_blankspaces_contingently()
        if row < 7: FEN = FEN + "-"
        row += 1
    return FEN

def test_all():
    # some testing:

    testFEN = "1B1K4-1p5N-7p-1qp5-n1P5-8-6k1-b7"
    testSeq = "_B_K_____p_____N_______p_qp_____n_P___________________k_b_______"

    assert FEN_to_seq(testFEN) == testSeq
    assert seq_to_FEN(testSeq) == testFEN

    test=seq_to_FEN(testSeq)
    print("attempt: " + test)
    print("correct: " + testFEN)
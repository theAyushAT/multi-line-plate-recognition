index = { "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "I":65,"J": 49, "K": 50, "L": 51, "M": 52, "N": 53,"O":66, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}

updated_idx = {}
count = 1
for key,value in index.items():
    updated_idx.update({key:count})
    count +=1

print(updated_idx)
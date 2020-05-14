# Nithilam Subbaian Submission for Project #2

class CellEntry:
    def __init__(self, left, right1 = None, right2 = None):
        self.left = left
        self.right1 = right1
        self.right2 = right2

def printParse(node):
    if node.right2 == None:
        parse_result = "["+ node.left +" "+ node.right1.left +"]"
    else:
        parse_result = "["+ node.left +" "+ printParse(node.right1) +" "+ printParse(node.right2) +"]"
    return parse_result

if __name__ == "__main__":

    CNF_filename = input("Enter the name of a text file specifying a context free grammar (CFG) in Chomsky normal form (CNF): ")
    consolidated_rules = {} # dictionary for easy access
    with open(CNF_filename,'r') as CNF_file:
        for line in CNF_file:
            line_words = line.replace("\n", "").split(" --> ")
            if line_words[1] in consolidated_rules:
                consolidated_rules[line_words[1]].append(line_words[0])
            else:
                consolidated_rules[line_words[1]]= [line_words[0]]

    while True:
        sentence_toParse = input("\n--> Enter sentence to parse or type 'quit' to leave program: ")
        if sentence_toParse == "quit":  exit()

        words = sentence_toParse.split()
        CKY_table = []
        word_count = len(words)
        for i in range(word_count):
            CKY_table.append([[] for x in range(word_count+1)])

        # based on implementation in textbook from Figure 13.5, 13.7
        for j in range(1, word_count+1, 1):
            word = words[j-1]

            if word in consolidated_rules:
                for rule in consolidated_rules[word]:
                    CKY_table[j-1][j].append(CellEntry(left=rule, right1 = CellEntry(left = word)))

                for i in range(j-2, -1, -1):
                    for k in range(i+1, j, 1):
                        for entry_b in CKY_table[i][k]:
                            for entry_c in CKY_table[k][j]:
                                left_search = entry_b.left +" "+ entry_c.left
                                if left_search in consolidated_rules:
                                    for left in consolidated_rules[left_search]:
                                        CKY_table[i][j].append(CellEntry(left=left, right1= entry_b, right2=entry_c))
        parseNo = 0
        for n in CKY_table[0][-1]:
            if n.left == "S":
                parseNo +=1
                print("Parse", str(parseNo), ":", printParse(n))
        if parseNo == 0: print("NO VALID PARSES\n")

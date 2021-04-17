import spacy

nlp = spacy.load("en_core_web_sm")

def is_generator_empty(generator):
    for element in generator:
        return False
    return True

# function 1
def get_dependency_paths_from_root(sentences):
    doc = nlp(sentences)  # get the doc object
    sentences = list(doc.sents)  # used to handle multiple sentences
    sentences_in_form_of_dependencies = []
    sentence_index = 0
    for sentence in sentences:  # iterate foreach sentence
        sentences_in_form_of_dependencies.append([])
        queue = [sentence.root]
        while len(queue) > 0:
            current_node = queue.pop(0)
            if current_node.is_alpha:
                sentences_in_form_of_dependencies[sentence_index].append(
                    current_node.dep_ + " ==> " + [child.dep_ + "->" + child.text for child in current_node.children].__str__())
                for child in current_node.children:
                    queue.append(child)
        sentence_index += 1
    return sentences_in_form_of_dependencies

dependency_paths_sentences = get_dependency_paths_from_root("I saw the man with the telescope. That was cool")
for dependency_paths in dependency_paths_sentences:
    for dependency in dependency_paths:
        print(dependency)
    print("\n\n")

def get_dependency_paths(sentences):
    doc = nlp(sentences)  # get the doc object
    sentences = list(doc.sents)  # used to handle multiple sentences
    sentences_in_form_of_dependencies = []
    sentence_index = 0
    for sentence in sentences:  # iterate foreach sentence
        sentences_in_form_of_dependencies.append([])
        token_index = 0
        for token in sentence:
            sentences_in_form_of_dependencies[sentence_index].append([])
            while token.head != token:
                sentences_in_form_of_dependencies[sentence_index][token_index].insert(0, token.text)
                sentences_in_form_of_dependencies[sentence_index][token_index].insert(0, token.dep_ + "->")
                token = token.head
            sentences_in_form_of_dependencies[sentence_index][token_index].insert(0, token.text)
            sentences_in_form_of_dependencies[sentence_index][token_index].insert(0, token.dep_ + "->")
            token_index += 1
        sentence_index += 1
    return sentences_in_form_of_dependencies

print("function 1")
dependency_paths_sentences = get_dependency_paths("I saw the man with the telescope. That was cool")
for dependency_paths in dependency_paths_sentences:
    for dependency in dependency_paths:
        print(dependency)
    print("\n\n")


# function 2
def get_subtree_of_dependents_given_a_token(sentences):
    doc = nlp(sentences)  # get the doc object
    sentences = list(doc.sents)  # used to handle multiple sentences
    subtrees = []
    sentence_index = 0 # init sentence index
    for sentence in sentences:  # iterate foreach sentence
        subtrees.append([])
        token_index = 0
        for token in sentence:
            subtrees[sentence_index].append([])
            queue = [token]
            while len(queue) > 0:
                current_node = queue.pop(0)
                if current_node.is_alpha:
                    subtrees[sentence_index][token_index].append(
                        current_node.text + " ==> " + [child.text for child in current_node.children].__str__())
                    for child in current_node.children:
                        queue.append(child)
            token_index += 1
        sentence_index += 1
    return subtrees

print("function 2")
subtrees_of_sentences = get_subtree_of_dependents_given_a_token('I saw the man with a telescope.')
for subtrees in subtrees_of_sentences:
    for subtree in subtrees:
        print(subtree)

def check_if_subtree(sentence, subtree):
    subtree = set(subtree)
    doc = nlp(sentence)  # get the doc object

    subtrees_array = []
    subtree_index = 0
    for token in doc:
        if not is_generator_empty(token.children):
            subtrees_array.append([])
            queue = [token]
            while len(queue) > 0:
                current_node = queue.pop(0)
                for child in current_node.children:
                    subtrees_array[subtree_index].append(child.text)
                    queue.append(child)
            subtree_index += 1
    for tree in subtrees_array:
        print(tree)
        tree = set(tree)
        if tree == subtree:
            return True
    return False

print("function 3")
print(check_if_subtree("I saw the man with a telescope.", ['I', 'with', '.', 'the', 'telescope', 'a']))


# function 4
def identirfy_head_of_a_span(span):
    return span.root

print("function 4")
doc = nlp("I saw the man with the telescope.")
span = doc[2:5]
print(identirfy_head_of_a_span(span))


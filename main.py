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
            sentences_in_form_of_dependencies[sentence_index].append(
                current_node.dep_ + " ==> " + [child.dep_ + "->" + child.text for child in current_node.children].__str__())
            for child in current_node.children:
                queue.append(child)
        sentence_index += 1
    return sentences_in_form_of_dependencies


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
                sentences_in_form_of_dependencies[sentence_index][token_index].insert(
                    0, token.text)
                sentences_in_form_of_dependencies[sentence_index][token_index].insert(
                    0, token.dep_ + "->")
                token = token.head
            sentences_in_form_of_dependencies[sentence_index][token_index].insert(
                0, token.text)
            sentences_in_form_of_dependencies[sentence_index][token_index].insert(
                0, token.dep_ + "->")
            token_index += 1
        sentence_index += 1
    return sentences_in_form_of_dependencies


# function 2
def get_subtree_of_dependents_given_a_token(sentences):
    doc = nlp(sentences)  # get the doc object
    sentences = list(doc.sents)  # used to handle multiple sentences
    subtrees = []
    sentence_index = 0  # init sentence index
    for sentence in sentences:  # iterate foreach sentence
        subtrees.append([])
        token_index = 0
        for token in sentence:
            subtrees[sentence_index].append([])
            queue = [token]
            while len(queue) > 0:
                current_node = queue.pop(0)
                subtrees[sentence_index][token_index].append(
                    current_node.text + " ==> " + [child.text for child in current_node.children].__str__())
                for child in current_node.children:
                    queue.append(child)
            token_index += 1
        sentence_index += 1
    return subtrees

# function 3


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
        tree = set(tree)
        if tree == subtree:
            return True
    return False

# alternative to function 4 with a span obj as input


def identirfy_head_of_a_span(span):
    return span.root


# function 4
def identify_head_of_a_span_in_form_of_string(span):
    doc = nlp(span)
    for token in doc:
        if token.dep_ == "ROOT":
            return token.text
    return None


# function 5
def get_parts_of_sentence(sentence):
    doc = nlp(sentence)
    root = None
    for token in doc:  # get root
        if token.dep_ == "ROOT":
            root = token
            break
    parts_of_sentence = {}
    for child in root.children:
        if child.dep_ == "dobj":
            parts_of_sentence["dobj"] = bfs(child)
        if child.dep_ == "nsubj":
            parts_of_sentence["nsubj"] = bfs(child)
        if child.dep_ == "iobj":
            parts_of_sentence["iobj"] = bfs(child)
    return parts_of_sentence


def bfs(token):
    tree_array = []
    queue = [token]
    while len(queue) > 0:
        current_node = queue.pop(0)
        tree_array.append(current_node)
        for child in current_node.children:
            queue.append(child)
    return tree_array


sentence = input("Write a sentence to analyze: ")

print("function 1")
dependency_paths_sentences = get_dependency_paths(
    sentence)
for dependency_paths in dependency_paths_sentences:
    for dependency in dependency_paths:
        print(dependency)
    print("\n\n")

print("function 2")
subtrees_of_sentences = get_subtree_of_dependents_given_a_token(
    sentence)
for subtrees in subtrees_of_sentences:
    for subtree in subtrees:
        print(subtree)
print("\n\n")

print("function 3")
print(check_if_subtree("I saw the man with a telescope.",
      ['I', 'with', '.', 'the', 'telescope', 'a']))
print("\n\n")

print("function 4")
doc = nlp("I saw the man with the telescope.")
span = doc[2:5]
span = input("Write a portion of a sentence to be analyzed: ")
print(identify_head_of_a_span_in_form_of_string(span))
print("\n\n")

print("function 5")
print(get_parts_of_sentence(sentence))

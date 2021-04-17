import spacy

nlp = spacy.load("en_core_web_sm")

# function 1
def get_dependency_paths(sentences):
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
                    current_node.dep_ + " ==> " + [child.dep_ for child in current_node.children].__str__())
                for child in current_node.children:
                    queue.append(child)
        sentence_index += 1
    return sentences_in_form_of_dependencies

print(get_dependency_paths("I saw the man with the telescope"))

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


print(get_subtree_of_dependents_given_a_token("I saw the man with the telescope. It was very interesting and funny at the same time"))

# function 4
def identirfy_head_of_a_span(span):
    return span.root

doc = nlp("I saw the man with the telescope.")
span = doc[2:5]
print(identirfy_head_of_a_span(span))

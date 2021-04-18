# NLU Assignment 1

## Student
* Name: Alessandro
* Surname: Grassi
* Student number: 221224

## Requirements

The main.py script requires python, spacy and its english models to be installed.

To install spacy through pip run the following command in the shell

```sh
pip install spacy
```

To install the english models run the following command in the shell

```sh
python -m spacy download en_core_web_sm
```

# Report

## function 1

Task: expract a path of dependency relations from the ROOT to a token

The task of the first function is to take one sentence as input and, for each token, return the path from the root to that token specifying what kind of dependency label the arcs.

Below can be seen an example where the token at the current iteration is a

```python
['ROOT->', 'saw', 'dobj->', 'man', 'prep->', 'with', 'pobj->', 'telescope', 'det->', 'a']
```

To achieve the result the input string is given as input to the parser, then it is created a list of sentences. The input of the function should be just a single sentence, but in case more sentences are provided they will be splitted and processed one by one. After that each sentence is iterated and for each sentence each token is iterated, the token and its dependency with its head are inserted in the list, this is done until the root token is reached, it will not be inserted so the root will be appended outside the loop.

```python
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
```

## function 2

Task: extract subtree of a dependents given a token

The algorithm splits each sentence if more then one is provided. For each sentence each token is added to the list of subtrees a list composed of the subtree of the token ordered as the original sentence
```python
def get_subtree_of_dependents_given_a_token(sentences):
    doc = nlp(sentences)  # get the doc object
    sentences = list(doc.sents)  # used to handle multiple sentences
    subtrees = []
    sentence_index = 0  # init sentence index
    for sentence in sentences:  # iterate foreach sentence
        subtrees.append([])
        token_index = 0
        for token in sentence:
            subtrees[sentence_index].append([subtree_token.text for subtree_token in token.subtree])
        sentence_index += 1
    return subtrees
```

## function 3

Task: check if a given list of tokens (segment of a sentence) forms a subtree

The task is to check if the dependency tree created starting from the input sentence has as a subtree the one given as input.

The algorithm works by ordering the given subtree, this is useful to perform the equality check. After that every token of the sentence is iterated. For each iteration the token is inserted in a list that will be treated as a queue, this is done to performe a bfs (Breadth First Search) though the graph. With this metod a tree is created for each token with that token a root node, after the tokens are done each tree is ordered and compared with the input one, if they match the algorithm return `true`

```python
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
```
Since there is no provided method to check if a generator is empty the below function check if a generator as elements in it
```python
def is_generator_empty(generator):
    for element in generator:
        return False
    return True
```

## function 4
Task: identify head of a span, given its tokens

The algorithm works by receiving a span composed in form of a `string`. Each token is iteretated untile the ROOT one is reached and will be returned.
```python
def identify_head_of_a_span_in_form_of_string(span):
    doc = nlp(span)
    for token in doc:
        if token.dep_ == "ROOT":
            return token.text
    return None
```

It is also provided a variant that takes a version of the algorithm that takes a span object as input
```python
def identirfy_head_of_a_span(span):
    return span.root
```

## function 5
Task: extract sentence subject, direct object and indirect object spans

The algorithm works by finding first the root token. Then its children are viewed and if their dependency match `dobj` or `nsubj` or `iobj` a bfs search is performed starting from that token.
```python
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
```

The code below performs a bfs search implemented with a list treated as a queue. Returns a list composed by all the nodes encountered including the starting one.
```python
def bfs(token):
    tree_array = []
    queue = [token]
    while len(queue) > 0:
        current_node = queue.pop(0)
        tree_array.append(current_node)
        for child in current_node.children:
            queue.append(child)
    return tree_array
```
# how to test
When the main.py script is executed in the shell you will be asket to write a sentence to analyze. Before the fourth function you will have to write a span to analyze, the output of the function is already provided in a readable way.
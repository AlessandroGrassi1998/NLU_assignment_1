import spacy
from nltk.corpus import dependency_treebank
from nltk.parse.transitionparser import *
from sklearn import tree


nlp = spacy.load("en_core_web_sm")


def bfs(token):
    tree_array = []
    queue = [token]
    while len(queue) > 0:
        current_node = queue.pop(0)
        tree_array.append(current_node)
        for child in current_node.children:
            queue.append(child)
    return tree_array


def is_generator_empty(generator):
    for element in generator:
        return False
    return True


# function 1
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
            subtrees[sentence_index].append(
                [subtree_token.text for subtree_token in token.subtree])
        sentence_index += 1
    return subtrees


# function 3
def check_if_subtree(sentence, subtree):
    subtree.sort()
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
        tree.sort()
        if tree == subtree:
            return True
    return False


# function 4
def identify_head_of_a_span_in_form_of_string(span):
    doc = nlp(span)
    for token in doc:
        if token.dep_ == "ROOT":
            return token.text
    return None

# alternative to function 4 with a span obj as input


def identirfy_head_of_a_span(span):
    return span.root


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
        if child.dep_ == "dative":
            parts_of_sentence["iobj"] = bfs(child)
    return parts_of_sentence


# sentence = input("Write a sentence to analyze: ")
sentence = input("Entry a sentence to be analyzed: ")

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
print(check_if_subtree("I saw the man with a telescope.",
      ['I', 'man', 'with', '.', 'the', 'telescope', 'a']))
print("\n\n")

print("function 4")
doc = nlp("I saw the man with the telescope.")
span = doc[2:5]
span = input("entry a span to be analyzed: ")
# span = input("Write a portion of a sentence to be analyzed: ")
print(identify_head_of_a_span_in_form_of_string(span))
print("\n\n")

print("function 5")
print(get_parts_of_sentence(sentence))


# optional and advanced part of the assignment
class ModifiedConfiguration(Configuration):
    def extract_features(self):
        result = []
        if len(self.stack) > 0:
            stack_idx0 = self.stack[len(self.stack) - 1]
            token = self._tokens[stack_idx0]
            if "head" in token and self._check_informative(token["head"]):
                result.append("STK_0_HEAD_" + str(token["head"]).upper())
            if "lemma" in token and self._check_informative(token["lemma"]):
                result.append("STK_0_LEMMA_" + token["lemma"].upper())
            if "tag" in token and self._check_informative(token["tag"]):
                result.append("STK_0_POS_" + token["tag"].upper())
            if "rel" in token and self._check_informative(token["rel"]):
                result.append("STK_0_REL_" + token["rel"].upper())
            if "deps" in token and token["deps"]:
                for d in token["deps"]:
                    result.append("STK_0_DEP_" + str(d).upper())
            if "feats" in token and self._check_informative(token["feats"]):
                feats = token["feats"].split("|")
                for feat in feats:
                    result.append("STK_0_FEATS_" + feat.upper())
            if len(self.stack) > 1:
                stack_idx1 = self.stack[len(self.stack) - 2]
                token = self._tokens[stack_idx1]
                if "head" in token and self._check_informative(token["head"]):
                    result.append("STK_1_HEAD_" + str(token["head"]).upper())
                if "lemma" in token and self._check_informative(token["lemma"]):
                    result.append("STK_1_LEMMA_" + token["lemma"].upper())
                if "rel" in token and self._check_informative(token["rel"]):
                    result.append("STK_1_REL_" + token["rel"].upper())
                if "deps" in token and token["deps"]:
                    for d in token["deps"]:
                        result.append("STK_1_DEP_" + str(d).upper())
                if "feats" in token and self._check_informative(token["feats"]):
                    feats = token["feats"].split("|")
                    for feat in feats:
                        result.append("STK_1_FEATS_" + feat.upper())
            if len(self.stack) > 2:
                stack_idx2 = self.stack[len(self.stack) - 3]
                token = self._tokens[stack_idx2]
                if self._check_informative(token["tag"]):
                    result.append("STK_2_POS_" + token["tag"].upper())

            left_most = 1000000
            right_most = -1
            dep_left_most = ""
            dep_right_most = ""
            for (wi, r, wj) in self.arcs:
                if wi == stack_idx0:
                    if (wj > wi) and (wj > right_most):
                        right_most = wj
                        dep_right_most = r
                    if (wj < wi) and (wj < left_most):
                        left_most = wj
                        dep_left_most = r
            if self._check_informative(dep_left_most):
                result.append("STK_0_LDEP_" + dep_left_most.upper())
            if self._check_informative(dep_right_most):
                result.append("STK_0_RDEP_" + dep_right_most.upper())

        if len(self.buffer) > 0:
            buffer_idx0 = self.buffer[0]
            token = self._tokens[buffer_idx0]
            if "head" in token and self._check_informative(token["head"]):
                result.append("BUF_0_HEAD_" + str(token["head"]).upper())
            if "lemma" in token and self._check_informative(token["lemma"]):
                result.append("BUF_0_LEMMA_" + token["lemma"].upper())
            if "tag" in token and self._check_informative(token["tag"]):
                result.append("BUF_0_POS_" + token["tag"].upper())
            if "rel" in token and self._check_informative(token["rel"]):
                result.append("BUF_0_REL_" + token["rel"].upper())
            if "deps" in token and token["deps"]:
                for d in token["deps"]:
                    result.append("BUF_0_DEP_" + str(d).upper())
            if "feats" in token and self._check_informative(token["feats"]):
                feats = token["feats"].split("|")
                for feat in feats:
                    result.append("BUF_0_FEATS_" + feat.upper())
            if len(self.buffer) > 1:
                buffer_idx1 = self.buffer[1]
                token = self._tokens[buffer_idx1]
                if "head" in token and self._check_informative(token["head"]):
                    result.append("BUF_1_HEAD_" + str(token["head"]).upper())
                if "lemma" in token and self._check_informative(token["lemma"]):
                    result.append("BUF_1_LEMMA_" + token["lemma"].upper())
                if "rel" in token and self._check_informative(token["rel"]):
                    result.append("BUF_1_REL_" + token["rel"].upper())
                if "deps" in token and token["deps"]:
                    for d in token["deps"]:
                        result.append("BUF_1_DEP_" + str(d).upper())
                if "feats" in token and self._check_informative(token["feats"]):
                    feats = token["feats"].split("|")
                    for feat in feats:
                        result.append("BUF_1_FEATS_" + feat.upper())
            if len(self.buffer) > 2:
                buffer_idx2 = self.buffer[2]
                token = self._tokens[buffer_idx2]
                if self._check_informative(token["tag"]):
                    result.append("BUF_2_POS_" + token["tag"].upper())

            left_most = 1000000
            right_most = -1
            dep_left_most = ""
            dep_right_most = ""
            for (wi, r, wj) in self.arcs:
                if wi == buffer_idx0:
                    if (wj > wi) and (wj > right_most):
                        right_most = wj
                        dep_right_most = r
                    if (wj < wi) and (wj < left_most):
                        left_most = wj
                        dep_left_most = r
            if self._check_informative(dep_left_most):
                result.append("BUF_0_LDEP_" + dep_left_most.upper())
            if self._check_informative(dep_right_most):
                result.append("BUF_0_RDEP_" + dep_right_most.upper())

        return result


class ModifyiedTransitionParser(TransitionParser):
    def _create_training_examples_arc_std(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
        operation = Transition(self.ARC_STANDARD)
        count_proj = 0
        training_seq = []

        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue

            count_proj += 1
            conf = ModifiedConfiguration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)

                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ":" + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        precondition = True
                        # Get the max-index of buffer
                        maxID = conf._max_address

                        for w in range(maxID + 1):
                            if w != b0:
                                relw = self._get_dep_relation(b0, w, depgraph)
                                if relw is not None:
                                    if (b0, relw, w) not in conf.arcs:
                                        precondition = False

                        if precondition:
                            key = Transition.RIGHT_ARC + ":" + rel
                            self._write_to_file(
                                key, binary_features, input_file)
                            operation.right_arc(conf, rel)
                            training_seq.append(key)
                            continue

                # Shift operation as the default
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)

            print(" Number of training examples : " + str(len(depgraphs)))
            print(" Number of valid (projective) examples : " + str(count_proj))
            return training_seq

    def _create_training_examples_arc_eager(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : 'A Dynamic Oracle for Arc-Eager Dependency Parsing' by Joav Goldberg and Joakim Nivre
        """
        operation = Transition(self.ARC_EAGER)
        countProj = 0
        training_seq = []

        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue

            countProj += 1
            conf = ModifiedConfiguration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(
                    features)

                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ":" + rel
                        self._write_to_file(
                            key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        key = Transition.RIGHT_ARC + ":" + rel
                        self._write_to_file(
                            key, binary_features, input_file)
                        operation.right_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # reduce operation
                    flag = False
                    for k in range(s0):
                        if self._get_dep_relation(k, b0, depgraph) is not None:
                            flag = True
                        if self._get_dep_relation(b0, k, depgraph) is not None:
                            flag = True
                    if flag:
                        key = Transition.REDUCE
                        self._write_to_file(
                            key, binary_features, input_file)
                        operation.reduce(conf)
                        training_seq.append(key)
                        continue

                # Shift operation as the default
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)

        print(" Number of training examples : " + str(len(depgraphs)))
        print(" Number of valid (projective) examples : " + str(countProj))
        return training_seq

    def parse(self, depgraphs, modelFile):
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        result = []
        # First load the model
        model = pickle.load(open(modelFile, "rb"))
        operation = Transition(self._algorithm)

        for depgraph in depgraphs:
            conf = ModifiedConfiguration(depgraph)
            while len(conf.buffer) > 0:
                features = conf.extract_features()
                col = []
                row = []
                data = []
                for feature in features:
                    if feature in self._dictionary:
                        col.append(self._dictionary[feature])
                        row.append(0)
                        data.append(1.0)
                np_col = array(sorted(col))  # NB : index must be sorted
                np_row = array(row)
                np_data = array(data)

                x_test = sparse.csr_matrix(
                    (np_data, (np_row, np_col)), shape=(
                        1, len(self._dictionary))
                )
                prob_dict = {}
                pred_prob = model.predict_proba(x_test)[0]
                for i in range(len(pred_prob)):
                    prob_dict[i] = pred_prob[i]
                sorted_Prob = sorted(
                    prob_dict.items(), key=itemgetter(1), reverse=True)

                # Note that SHIFT is always a valid operation
                for (y_pred_idx, confidence) in sorted_Prob:
                    # y_pred = model.predict(x_test)[0]
                    # From the prediction match to the operation
                    y_pred = model.classes_[y_pred_idx]

                    if y_pred in self._match_transition:
                        strTransition = self._match_transition[y_pred]
                        baseTransition = strTransition.split(":")[0]

                        if baseTransition == Transition.LEFT_ARC:
                            if (
                                operation.left_arc(
                                    conf, strTransition.split(":")[1])
                                != -1
                            ):
                                break
                        elif baseTransition == Transition.RIGHT_ARC:
                            if (
                                operation.right_arc(
                                    conf, strTransition.split(":")[1])
                                != -1
                            ):
                                break
                        elif baseTransition == Transition.REDUCE:
                            if operation.reduce(conf) != -1:
                                break
                        elif baseTransition == Transition.SHIFT:
                            if operation.shift(conf) != -1:
                                break
                    else:
                        raise ValueError(
                            "The predicted transition is not recognized, expected errors"
                        )

            # Finish with operations build the dependency graph from Conf.arcs

            new_depgraph = deepcopy(depgraph)
            for key in new_depgraph.nodes:
                node = new_depgraph.nodes[key]
                node["rel"] = ""
                # With the default, all the token depend on the Root
                node["head"] = 0
            for (head, rel, child) in conf.arcs:
                c_node = new_depgraph.nodes[child]
                c_node["head"] = head
                c_node["rel"] = rel
            result.append(new_depgraph)

        return result

    def train(self, depgraphs, modelfile, verbose=True):
        """
        :param depgraphs : list of DependencyGraph as the training data
        :type depgraphs : DependencyGraph
        :param modelfile : file name to save the trained model
        :type modelfile : str
        """
        try:
            input_file = tempfile.NamedTemporaryFile(
                prefix="transition_parse.train", dir=tempfile.gettempdir(), delete=False
            )

            if self._algorithm == self.ARC_STANDARD:
                self._create_training_examples_arc_std(depgraphs, input_file)
            else:
                self._create_training_examples_arc_eager(depgraphs, input_file)

            input_file.close()
            # Using the temporary file to train the libsvm classifier
            x_train, y_train = load_svmlight_file(input_file.name)
            # The parameter is set according to the paper:
            # Algorithms for Deterministic Incremental Dependency Parsing by Joakim Nivre
            # Todo : because of probability = True => very slow due to
            # cross-validation. Need to improve the speed here
            clf = tree.DecisionTreeClassifier()
            clf.fit(x_train, y_train)
            pickle.dump(clf, open(modelfile, "wb"))
            """model = svm.SVC(
                kernel="poly",
                degree=2,
                coef0=0,
                gamma=0.2,
                C=0.5,
                verbose=verbose,
                probability=True,
            )

            model.fit(x_train, y_train)
            # Save the model to file name (as pickle)
            pickle.dump(model, open(modelfile, "wb"))"""
        finally:
            remove(input_file.name)


# Evaluate the features comparing performance to the original
print()
print()
print("Optional and advanced part")
print()
transition_parser = TransitionParser("arc-standard")
transition_parser.train(dependency_treebank.parsed_sents()[
                        :100], "transition_parser.model")
parses = transition_parser.parse(
    dependency_treebank.parsed_sents()[-10:], "transition_parser.model")
print(len(parses))
dependency_evaluator = DependencyEvaluator(
    parses, dependency_treebank.parsed_sents()[-10:])
standard_parser_evaluation = dependency_evaluator.eval()
print(
    f"The scores of the standard TransitionParser are: {standard_parser_evaluation}")
print()


my_transition_parser = ModifyiedTransitionParser("arc-standard")
my_transition_parser.train(dependency_treebank.parsed_sents()[
                           :100], "modifyied_transition_parser.model")
parses = my_transition_parser.parse(
    dependency_treebank.parsed_sents()[-10:], "modifyied_transition_parser.model")
print(len(parses))
dependency_evaluator = DependencyEvaluator(
    parses, dependency_treebank.parsed_sents()[-10:])
modifyied_parser_evaluation = dependency_evaluator.eval()
print(
    f"The scores of ModifyiedTransitionParser are: {dependency_evaluator.eval()}")
print()
print("The modifyied parser has a difference of: " + str((modifyied_parser_evaluation[0] - standard_parser_evaluation[0])) + " " + str((
    modifyied_parser_evaluation[1] - standard_parser_evaluation[1])))

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MNLI_LABEL = ['entailment', 'neutral', 'contradiction',
              'entailment\'', 'neutral\'', 'contradiction\'']
EQ_LABEL = ['equivalent', 'not_equivalent', 'equivalent\'', 'not_equivalent\'']
ENTAIL_LABEL = ['entailment', 'not_entailment', 'entailment\'',
                'not_entailment\'', '0', '1', '0\'', '1\'']

LABEL_SET = {
    'sst2': ['positive', 'negative', 'positive\'', 'negative\'', '0', '1', '0\'', '1\''],
    'mnli': MNLI_LABEL,
    'mnli_mismatched': MNLI_LABEL,
    'mnli_matched': MNLI_LABEL,
    "anli_r1": MNLI_LABEL,
    "anli_r2": MNLI_LABEL,
    "anli_r3": MNLI_LABEL,
    "snli": MNLI_LABEL,
    'qqp': EQ_LABEL,
    'qnli': ENTAIL_LABEL,
    'rte': ENTAIL_LABEL,
    'cola': ['unacceptable', 'acceptable', 'unacceptable\'', 'acceptable\''],
    'mrpc': EQ_LABEL,
    "paws":EQ_LABEL,
    'wnli': ENTAIL_LABEL,
}

MODEL_SET = [
    'contrastive_llama',    # This project supports contrastive_llama only
    'google/flan-t5-large',
    'EleutherAI/gpt-neox-20b',
    'tiiuae/falcon-40b-instruct',
    'llama-13b',
    'llama2-13b',
    'llama2-13b-chat',
    'llama2-7b',
    'llama2-7b-chat',
    'vicuna-13b',
    'vicuna-13b-v1.3',
    'google/flan-ul2',
    'cerebras/Cerebras-GPT-13B',
    'databricks/dolly-v1-6b',
    'chatgpt',
    'gpt4',
]

LABEL_TO_ID = {
    'sst2': {'negative': 0, 'positive': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'mnli_mismatched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'mnli_matched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'anli_r1': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'anli_r2': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'anli_r3': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'snli': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2},
    'qqp': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1},
    'qnli': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'rte': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'cola': {'unacceptable': 0, 'acceptable': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mrpc': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1},
    'paws': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1},
    'wnli': {'entailment': 1, 'not_entailment': 0, '0': 0, '1': 1, 0: 0, 1: 1},
}

ID_TO_LABEL = {
    'sst2': {0: 'negative', 1: 'positive'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_matched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_mismatched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'anli_r1': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'anli_r2': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'anli_r3': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'snli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'qqp': {1: 'equivalent', 0: 'not_equivalent'},
    'qnli': {0: 'entailment', 1: 'not_entailment'},
    'rte': {0: 'entailment', 1: 'not_entailment'},
    'cola': {0: 'unacceptable', 1: 'acceptable'},
    'mrpc': {1: 'equivalent', 0: 'not_equivalent'},
    'paws': {1: 'equivalent', 0: 'not_equivalent'},
    'wnli': {1: 'entailment', 0: 'not_entailment'},
}

# Randomly generated sequences for CheckList attack
CHECKLIST_RANDOM_SEQ = ['LGOZMPXsPd', 'DIHVWPN6U1', 'lhKrdEwcN5', 'sbkysCbk66', '1xD6X68vTP', 'udfR3V237Z',
 '9DDjwFpttG', '4SOKVvzB2G', '01kgeIBb1R', 'zkMDpiXzU2', 'XPY8pzwr1o', 'V87xnYBCWj',
 '4O0NzsP7eH', 'WbDVZyVp1E', 'W22SuitsNN', '5AOmoyDeLq', 'H8LaXn4Hg8', 'l9LJthZz1b',
 '4wLJkOiOOg', 'C5fJuobIC2', 'yMTNZJiQw9', '3v25o8DEaX', 'mjQn1JNm0F', 'XMGlAQPLOu',
 'KZqJefM6uA', 'IwjWHWnBSY', 'luaZjkJZxT', '9mADXFVHmL', 'FNwbdoBhxw', 'vM5mCdHmRc',
 'zPemJFN4EL', '25dqGJsl9E', 'W1PrUIXVep', 'Exq4dQc7Gu', '1D0S92CHZn', 'b1enF1lZju',
 'iMl9M43T9G', 'wrStVPqjdo', 'tR2NEOOGc6', 'FrUdx2vEP4', '9Kk6NIONDS', 'ICHqkJRIGN',
 'tDwBD1JEMK', 'AUMrUVMXTY', 'bAtiKPGaqS', 'v0weCX28gl', '3hjOlsH94W', 'EqSCSFaq9b',
 'Iyk1lJt8yw', 'aaQybJqQOz', '2Ma2zSrlbU', 'Mr9LRn9VET', 'G49exb0wnm', '9hDFyMoXgX',
 'bg0UJsr9dR', 'mhpzd91wDh', 'rnG1w9Td1V', '5sKrXp8HGm', 'Yq4I8aBRkn', 'IXJCQn7CRO',
 '9LbBv68VVd', 'LjU1kRjJlX', 'JUI83rAewS', '9BlumhJ9sT', 'Ee5npqYDtm', 'iDDHPq06uH',
 'rivFsYFkix', 'InBRuQ7RrQ', 'CKs02aCQgF', '9FkhdI9r16', 'le65w5L9DV', 'T5A0HhW1cy']

# Perturbed instructions for evaluation. Each task has 6 prompts for each type of perturbation.
PROMPT_SET_Promptbench_adv = {
    'clean': {
        'sst2': [
            "Read the provided excerpt and choose between 'positive' and 'negative' to describe its sentiment: ",
            "Analyze the tone of this statement and respond with either 'positive' or 'negative': ",
            "Evaluate the sentiment of the given text and classify it as 'positive' or 'negative': ",
            "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: ",
            "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to classify this statement: ",
            "Acting as a sentiment evaluator, identify if the given sentence is 'positive' or 'negative'. Classify: ",
        ],
        'qqp': [
            "Can these two questions be considered equal in meaning? Answer with 'equivalent' or 'not_equivalent': ",
            'Are the following two questions equivalent or not? Answer me with "equivalent" or "not_equivalent". ',
            "Determine if the given pair of statements can be considered the same by responding with 'equivalent' or 'not_equivalent'. ",
            "In your role as a question comparison tool, assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent'. ",
            "As a question equivalence detection system, examine the provided questions and respond with 'equivalent' if they are the same in meaning, or 'not_equivalent' if they are different. ",
            "Functioning as a question similarity evaluation tool, analyze the given questions and decide if they share the same meaning, responding with 'equivalent' or 'not_equivalent'. ",
        ],
        'mnli': [
            "Assess the connection between the following sentences and classify it as 'entailment', 'neutral', or 'contradiction': ",
            "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':",
            "Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. Answer with either 'entailment', 'neutral', or 'contradiction':",
            "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction':",
            "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment', 'neutral', or 'contradiction':",
            "Operate as a tool to assess entailment by examining the given sentences and determine whether their relationship is 'entailment', 'neutral', or 'contradiction'.",
        ],
        'qnli': [
            "Consider the context and question, and indicate if the answer can be logically deduced from the context by responding with 'entailment' or 'not_entailment'.",
            "Given the question and context provided, determine if the answer can be inferred by choosing 'entailment' or 'not_entailment'. ",
            "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'. ",
            "As a language expert, assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment'. ",
            "In your role as a semantic evaluator, determine if the provided context justifies the answer to the question and answer with 'entailment' or 'not_entailment'. ",
            "As a textual analyst, examine if the given context logically implies the answer to the question and indicate your decision with 'entailment' or 'not_entailment'. ",
        ],
        'rte': [
            "Determine if the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment'. ",
            'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment", just one word. ',
            "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment'.",
            "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment':",
            "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment':",
            "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment':",
        ],
        'cola': [
            "Review the sentence below and identify whether its grammar is 'Acceptable' or 'Unacceptable': ",
            "Assess the following sentence and determine if it is grammatically correct. Respond with 'Acceptable' or 'Unacceptable':",
            "Examine the given sentence and decide if it is grammatically sound. Answer with either 'Acceptable' or 'Unacceptable':",
            "In your role as a grammar check tool, assess the following sentence and classify it as 'acceptable' if it is grammatically correct or 'unacceptable' if it is incorrect:",
            "As a grammar identification system, examine the provided sentence and respond with 'acceptable' for grammatically correct sentences or 'unacceptable' for incorrect ones:",
            "Functioning as a grammar evaluation tool, analyze the given sentence and decide if it is grammatically correct, responding with 'acceptable' or 'unacceptable':",
        ],
        'mrpc': [
            "Can the given sentences be considered semantically identical? Please reply with 'equivalent' or 'not_equivalent'. ",
            "Do these two sentences have the same underlying meaning? Respond with 'equivalent' or 'not_equivalent'. ",
            "Are the meanings of the following pair of sentences the same? Answer with 'equivalent' or 'not_equivalent'. ",
            "As a semantic comparison expert, evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent'. ",
            "In your capacity as a language analyst, assess the following sentences and classify their similarity as 'equivalent' or 'not_equivalent'. ",
            "As a sentence similarity evaluator, analyze the provided sentences and indicate if their meanings are 'equivalent' or 'not_equivalent'. ",
        ],
        'wnli': [
            "Identify whether the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment'. ",
            'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment", just one word. ',
            "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment'.",
            "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment':",
            "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment':",
            "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment':",
        ],
        'squad_v2': [
            "Discover the best answer based on the context. If the context doesn't include an answer, respond with 'unanswerable'. ",
            "Based on the given context, provide the best possible answer. If there's no answer available in the context, respond with 'unanswerable'.",
            "Identify the most relevant answer from the context. If it's not possible to find an answer, respond with 'unanswerable'.",
            "As a well-informed specialist familiar with the context, provide an answer to the question. If the context doesn't contain an answer, reply with 'unanswerable'.",
            "Drawing upon your expertise in the context, determine the most suitable answer. If an answer isn't available, state 'unanswerable'.",
            "As a subject matter expert, extract the correct answer from the context. If an answer is not present, indicate 'unanswerable'.",
        ]
    },
    "checklist": {
        "cola": [
          "Review the sentence below and identify whether its grammar is 'Acceptable' or 'Unacceptable' LGOZMPXsPd: ",
          "Assess the following sentence and determine if it is grammatically correct. Respond with 'Acceptable' or 'Unacceptable' DIHVWPN6U1: ",
          "Examine the given sentence and decide if it is grammatically sound. Answer with either 'Acceptable' or 'Unacceptable' lhKrdEwcN5: ",
          "In your role as a grammar check tool, assess the following sentence and classify it as 'acceptable' if it is grammatically correct or 'unacceptable' if it is incorrect sbkysCbk66: ",
          "As a grammar identification system, examine the provided sentence and respond with 'acceptable' for grammatically correct sentences or 'unacceptable' for incorrect ones 1xD6X68vTP: ",
          "Functioning as a grammar evaluation tool, analyze the given sentence and decide if it is grammatically correct, responding with 'acceptable' or 'unacceptable' udfR3V237Z: "
        ],
        "mnli": [
          "Assess the connection between the following sentences and classify it as 'entailment', 'neutral', or 'contradiction' 9DDjwFpttG: ",
          "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction' 4SOKVvzB2G: ",
          "Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. Answer with either 'entailment', 'neutral', or 'contradiction' 01kgeIBb1R: ",
          "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction' zkMDpiXzU2: ",
          "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment', 'neutral', or 'contradiction' XPY8pzwr1o: ",
          "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment', 'neutral', or 'contradiction' V87xnYBCWj: "
        ],
        "mrpc": [
          "Can the given sentences be considered semantically identical? Please reply with 'equivalent' or 'not_equivalent' KZqJefM6uA. ",
          "Do these two sentences have the same underlying meaning? Respond with 'equivalent' or 'not_equivalent' IwjWHWnBSY. ",
          "Are the meanings of the following pair of sentences the same? Answer with 'equivalent' or 'not_equivalent' luaZjkJZxT. ",
          "As a semantic comparison expert, evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent' 9mADXFVHmL. ",
          "In your capacity as a language analyst, assess the following sentences and classify their similarity as 'equivalent' or 'not_equivalent' FNwbdoBhxw. ",
          "As a sentence similarity evaluator, analyze the provided sentences and indicate if their meanings are 'equivalent' or 'not_equivalent' vM5mCdHmRc. "
        ],
        "qnli": [
          "Consider the context and question, and indicate if the answer can be logically deduced from the context by responding with 'entailment' or 'not_entailment' zPemJFN4EL. ",
          "Given the question and context provided, determine if the answer can be inferred by choosing 'entailment' or 'not_entailment' 25dqGJsl9E. ",
          "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment' W1PrUIXVep. ",
          "As a language expert, assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment' Exq4dQc7Gu. ",
          "In your role as a semantic evaluator, determine if the provided context justifies the answer to the question and answer with 'entailment' or 'not_entailment' 1D0S92CHZn. ",
          "As a textual analyst, examine if the given context logically implies the answer to the question and indicate your decision with 'entailment' or 'not_entailment' b1enF1lZju. "
        ],
        "qqp": [
          "Can these two questions be considered equal in meaning? Answer with 'equivalent' or 'not_equivalent' iMl9M43T9G. ",
          "Are the following two questions equivalent or not? Answer me with \"equivalent\" or \"not_equivalent\" wrStVPqjdo. ",
          "Determine if the given pair of statements can be considered the same by responding with 'equivalent' or 'not_equivalent' tR2NEOOGc6. ",
          "In your role as a question comparison tool, assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent' FrUdx2vEP4. ",
          "As a question equivalence detection system, examine the provided questions and respond with 'equivalent' if they are the same in meaning, or 'not_equivalent' if they are different 9Kk6NIONDS. ",
          "Functioning as a question similarity evaluation tool, analyze the given questions and decide if they share the same meaning, responding with 'equivalent' or 'not_equivalent' ICHqkJRIGN. "
        ],
        "rte": [
          "Determine if the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment' tDwBD1JEMK. ",
          "Are the following two sentences entailment or not_entailment? Answer me with \"entailment\" or \"not_entailment\", just one word AUMrUVMXTY. ",
          "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment' bAtiKPGaqS. ",
          "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment' v0weCX28gl: ",
          "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment' 3hjOlsH94W: ",
          "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment' EqSCSFaq9b: "
        ],
        "sst2": [
          "Read the provided excerpt and choose between 'positive' and 'negative' to describe its sentiment Iyk1lJt8yw: ",
          "Analyze the tone of this statement and respond with either 'positive' or 'negative' aaQybJqQOz: ",
          "Evaluate the sentiment of the given text and classify it as 'positive' or 'negative' 2Ma2zSrlbU: ",
          "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify Mr9LRn9VET: ",
          "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to classify this statement G49exb0wnm: ",
          "Acting as a sentiment evaluator, identify if the given sentence is 'positive' or 'negative'. Classify 9hDFyMoXgX: "
        ],
        "wnli": [
          "Identify whether the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment' bg0UJsr9dR. ",
          "Are the following two sentences entailment or not_entailment? Answer me with \"entailment\" or \"not_entailment\", just one word mhpzd91wDh. ",
          "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment' rnG1w9Td1V. ",
          "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment' 5sKrXp8HGm: ",
          "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment' Yq4I8aBRkn: ",
          "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment' IXJCQn7CRO: "
        ]
      },
    "deepwordbug": {
        "cola": [
          "Reiew the seVntence below and identifpy wheoher its gVammar is 'Acceptable' or 'Unacceptable': ",
          "Asess the foBllowing senence and defermine if it is grammaGtically cRrrect. ResMond with 'Acceptable' or 'Unacceptable':",
          "EJamine the givPen senence and decdie if it is gammatically souZnd. Aaswer with eZther 'Acceptable' or 'Unacceptable':",
          "In your role as a grammr ceck gool, assegss the followin entence and cclassify it as 'acceptable' if it is Dgrammatically corrIct or 'unacceptable' if it is icnorrect:",
          "A a grammNar identifciation syTstem, examie the povided senteXce and erspond with 'acceptable' for grammaticalgy corDrect sentOnces or 'unacceptable' for inckrrect oens:",
          "FuncQioning as a gRrammar eevaluation tol, Ianalyze the giUen esntence and decidy if it is gRammatically coorrect, Responding with 'acceptable' or 'unacceptable':"
        ],
        "mnli": [
          "AXssess the connedtion between the followign senteznces and clarsify it as 'entailment', 'neutral', or 'contradiction': ",
          "DKoes the reationship between the giveT entences repersent entailment, neutral, or contradiction? Repond with 'entailment', 'neutral', or 'contradiction':",
          "Exaimne the air of entences and dItermine if they eVhibit entailment, neutral, or contradiction. Ansewr with eitehr 'entailment', 'neutral', or 'contradiction':",
          "Given your rWole as an entailment analysid otol, asess the rBelationship between the gMven senences and Mlassify it as 'entailment', 'neutral', or 'contradiction':",
          "As an entailment identifciation syUstem, eamine the conneRction between the foIlowing Pentences and resphnd with 'entailment', 'neutral', or 'contradiction':",
          "Bunctioning as an entailment evDluation Utool, anaGlyze the provWded sentenecs and dOcide if their rGelationship is 'entailment', 'neutral', or 'contradiction':"
        ],
        "mrpc": [
          "Cfan the gien seentences be condsidered semanticaly dientical? PRease reply with 'equivalent' or 'not_equivalent'. ",
          "Do these Lwo esntences have the same nderlying meanLing? Respnod with 'equivalent' or 'not_equivalent'. ",
          "HAre the maenings of the foQllowing apir of sentenes the same? Anser with 'equivalent' or 'not_equivalent'. ",
          "As a Dsemantic comparison eaxpert, evDluate the ggiven paIir of senGtences and determiine if they are 'equivalent' or 'not_equivalent'. ",
          "In your caacity as a langauge analyst, asess the fNllowing sentencHes and classifQy their siJilarity as 'equivalent' or 'not_equivalent'. ",
          "As a sentince gsimilarity Jevaluator, ajalyze the providMed Tsentences and idnicate if their emanings are 'equivalent' or 'not_equivalent'. "
        ],
        "qnli": [
          "Cnsider the cdntext and quBestion, and indicae if the anscwer can be lagically deduceX from the cntext by kesponding with 'entailment' or 'not_entailment'.",
          "GivCn the queBstion and cnotext porvided, dStermine if the acnswer can be infLerred by chjoosing 'entailment' or 'not_entailment'. ",
          "BaBed on the crovided contet and questcon, deciRde if the iQnformation supgorts the anJswer by rAsponding with 'entailment' or 'not_entailment'. ",
          "As a langauge Zexpert, Aassess if the gien coVtext eYtails the ansZer to the qeustion and espond with 'entailment' or 'not_entailment'. ",
          "In your rgole as a semanctic eEaluator, deDtermine if the provide conYext justbifies the anmswer to the questoin and answrr with 'entailment' or 'not_entailment'. ",
          "As a textual analys, Iexamine if the gibven conteht logLically impies the ansger to the Buestion and indicYate your dBecision with 'entailment' or 'not_entailment'. "
        ],
        "qqp": [
          "Can these two quesions be cconsidered equaal in maaning? Answre with 'equivalent' or 'not_equivalent': ",
          "ARe the fllowing mtwo Bquestions equivalent or not? Aeswer me with 'equivalent' or 'not_equivalent': ",
          "Deetrmine if the gievn pUair of stateGments can be consideed the same by respAnding with 'equivalent' or 'not_equivalent'. ",
          "In your rolle as a qusetion cAmparison Rtool, asVess the followwing paiir of questions and calssify them as 'equivalent' or 'not_equivalent'. ",
          "As a Yuestion Pequivalence deDtection syste, Rxamine the provide Yuestions and resond with 'equivalent' if they are the same in maening, or 'not_equivalent' if they are differeGnt. ",
          "Fuctioning as a qeustion ismilarity evaluaiton tool, analymze the gibven qestions and decEde if they sahre the same emaning, rKesponding with 'equivalent' or 'not_equivalent'. "
        ],
        "rte": [
          "DTetermine if the gievn pFir of esntences dPsplays entailment or not_entailment. Raspond with 'entailment' or 'not_entailment'. ",
          "As the fgollowing ltwo senteces entailment or not_entailment? ADnswer me with 'entailment' or 'not_entailment'. ",
          "DDoes the relaitonship between the giveD senetnces repreent entailment or not_entailment? NRespond with 'entailment' or 'not_entailment'.",
          "In your Bole as an entailment aalysis toEol, asCess the realtionship between the giOen sentenXes and cHassify it as 'entailment' or 'not_entailment':",
          "As an entailment dientification srstem, examiOe the connectiIon between the flolowing entences and orespond with 'entailment' or 'not_entailment':",
          "FSunctioning as an entailment eTvaluation nool, Aanalyze the Iprovided sentencAes and decidne if their relatioAship is 'entailment' or 'not_entailment':"
        ],
        "sst2": [
          "Read the provided ecxerpt and cEhoose between 'positive' and 'negative' to descrabe its sentimeWnt: ",
          "Aanlyze the tne of this satement and erspond with eFither 'positive' or 'negative': ",
          "Evaloate the sentimBnt of the biven ext and cZlassify it as 'positive' or 'negative': ",
          "As a sektiment calssifier, Ietermine hether the folQlowing bext is 'positive' or 'negative'. Plaese calssify: ",
          "In the orle of a esntiment anaDysis ool, resopnd with 'positive' or 'negative' to classAify this sqatement: ",
          "Acing as a seIntiment edaluator, dientify if the Xgiven esntence is 'positive' or 'negative'. Cblassify: "
        ],
        "wnli": [
          "Identmify whQther the giben air of sentence demonstrats entailment or not_entailment. SAnswer with 'entailment' or 'not_entailment'. ",
          "Are the follkowing tMo entences entailment or not_entailment? Akswer me with 'entailment' or 'not_entailment'. ",
          "Deos the relaitonship between the giRven sentence repTesent entailment or not_entailment? ROespond with 'entailment' or 'not_entailment'.",
          "In your orle as an entailment aalysis Mool, asYsess the elationship between the gcven senteIces and Qlassify it as 'entailment' or 'not_entailment':",
          "As an entailment iDdentification ssytem, examPne the cAonnection between the Following entences and reBpond with 'entailment' or 'not_entailment':",
          "FunctHioning as an entailment evaluaion Ztool, Inalyze the providDed senBences and deLide if their relaionship is 'entailment' or 'not_entailment':"
        ]
      },
    "textfooler": {
        "cola": [
          "Analyzed the assertion below and ascertain whether its grammar is 'Acceptable' or 'Unacceptable': ",
          "Evaluate the afterward sentence and ascertain if it is grammatically accurate. Answer with 'Acceptable' or 'Unacceptable':",
          "Analysed the gave sentence and chose if it is grammatically acoustic. Address with either 'Acceptable' or 'Unacceptable':",
          "In your role as a grammar check apparatus, evaluate the afterward sentence and categories it as 'acceptable' if it is grammatically accuracy or 'unacceptable' if it is amiss:",
          "As a grammar ascertain system, analyse the afforded sentence and accommodate with 'acceptable' for grammatically accuracy sentence or 'unacceptable' for amiss ones:",
          "Acting as a language analyses apparatus, analyse the gave sentence and ascertain if it is grammatically accuracy, answer with 'acceptable' or 'unacceptable':"
        ],
        "mnli": [
          "Appraisal the connected between the later sentences and categories it as 'entailment', 'neutral', or 'contradiction': ",
          "Could the connections between the gave sentences be entailment, neutral, or contradiction? Reacting with 'entailment', 'neutral', or 'contradiction':",
          "Analysed the couple of sentences and ascertain if they apiece entailment, neutral, or contradiction. Address with either 'entailment', 'neutral', or 'contradiction':",
          "About your feature as an entailment analyse appliance, appraisal the relationship between the gave sentences and categories it as 'entailment', 'neutral', or 'contradiction':",
          "As an entailment determining system, analyse the attach between the next sentences and answer with 'entailment', 'neutral', or 'contradiction':",
          "Acting as an entailment analyses apparatus, analyze the afforded sentences and choose if their affairs is 'entailment', 'neutral', or 'contradiction':"
        ],
        "mrpc": [
          "Kan the gave sentences are dealt semantically akin? Ask answered with 'equivalent' or 'not_equivalent'. ",
          "Do these two sentences have the same main connotation? Accommodate with 'equivalent' or 'not_equivalent'. ",
          "Are the connotation of the farther couple of sentences the same? Answered with 'equivalent' or 'not_equivalent'. ",
          "As a semantic comparative expertise, appraised the gave couple of sentences and ascertain if they are 'equivalent' or 'not_equivalent'. ",
          "In your abilities as a grammar commentator, assess the afterward sentences and categorize their analogy as 'equivalent' or 'not_equivalent'. ",
          "As a sentences likeness evaluator, analyze the afforded sentences and clarified if their connotation are 'equivalent' or 'not_equivalent'. "
        ],
        "qnli": [
          "Analyzed the context and topics, and clarified if the answer can are intelligently alleged from the context by answer with 'entailment' or 'not_entailment'.",
          "Accorded the issue and context afforded, ascertained if the answered can are alleged by choice 'entailment' or 'not_entailment'. ",
          "Anchored on the awarded context and issue, decide if the data aid the answer by cope with 'entailment' or 'not_entailment'. ",
          "As a linguistic expertise, appraisal if the allocated context assumes the address to the issue and accommodate with 'entailment' or 'not_entailment'. ",
          "In your feature as a semantic evaluator, ascertain if the awarded context deserve the answered to the issues and address with 'entailment' or 'not_entailment'. ",
          "As a textual commentator, analyse if the allocated context aptly assume the answered to the issue and clarified your decide with 'entailment' or 'not_entailment'. "
        ],
        "qqp": [
          "Can these two questions be analyzed same in connotation? Address with 'equivalent' or 'not_equivalent': ",
          "Are the below two questions same or not? Address me with 'equivalent' or 'not_equivalent'. ",
          "Ascertained if the given couple of statements can are analyzed the same by answer with 'equivalent' or 'not_equivalent'. ",
          "In your feature as a statement compare apparatus, appraisal the afterward coupled of questions and categories them as 'equivalent' or 'not_equivalent'. ",
          "As a statement equivalence detect system, analyse the given questions and answer with 'equivalent' if they are the same in connotation, or 'not_equivalent' if they are assorted. ",
          "Activities as a statement likeness analyses apparatus, analyse the gave questions and choose if they exchanging the same meaning, answer with 'equivalent' or 'not_equivalent'. "
        ],
        "rte": [
          "Deciding if the gave couples of sentence appear entailment or not_entailment. Replying with 'entailment' or 'not_entailment'. ",
          "Are the aftermath two sentences entailment or not_entailment? Answered me with 'entailment' or 'not_entailment'. ",
          "Ca the affairs between the made sentences pose entailment or not_entailment? React with 'entailment' or 'not_entailment'.",
          "In your feature as an entailment analyse apparatus, appraisal the affairs between the gave sentences and categories it as 'entailment' or 'not_entailment':",
          "As an entailment ascertain mechanism, analyse the attach between the afterward sentences and answer with 'entailment' or 'not_entailment':",
          "Acting as an entailment analyses apparatus, analyze the afforded sentences and ascertain if their relation is 'entailment' or 'not_entailment':"
        ],
        "sst2": [
          "Lea the afforded excerpts and choose between 'positive' and 'negative' to blueprint its sentiment: ",
          "Analyse the hue of this statement and accommodate with either 'positive' or 'negative': ",
          "Appraisals the sentiment of the allocated laws and categories it as 'positive' or 'negative': ",
          "As a sentiment classifier, ascertain whether the later laws is 'positive' or 'negative'. Please categories: ",
          "In the feature of a sentiment analyse apparatus, accommodate with 'positive' or 'negative' to categories this statement: ",
          "Behaving as a sentiment evaluator, ascertain if the gave sentence is 'positive' or 'negative'. Categories: "
        ],
        "wnli": [
          "Ascertain whether the gave couple of phrases attest entailment or not_entailment. Answering with 'entailment' or 'not_entailment'. ",
          "Are the aftermath two sentences entailment or not_entailment? Address me with 'entailment' or 'not_entailment'. ",
          "Ca the affairs between the gave sentences accounted entailment or not_entailment? Answer with 'entailment' or 'not_entailment'.",
          "En your feature as an entailment analysed apparatus, appraisals the relationship between the gave sentences and categories it as 'entailment' or 'not_entailment':",
          "As an entailment ascertain mechanism, analyse the connection between the afterward sentences and answer with 'entailment' or 'not_entailment':",
          "Roles as an entailment assess device, analyze the envisioned sentences and ascertain if their affairs is 'entailment' or 'not_entailment':"
        ]
      },
    'semantic': {
        'sst2': [
            "Examine the provided excerpt and select 'positive' or 'negative' to describe its sentiment: ",
            "Assess the tone of this statement and reply with either 'positive' or 'negative': ",
            "Assess the sentiment of the provided text and categorize it as 'positive' or 'negative': ",
            "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please categorize: ",
            "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to categorize this statement: ",
            "Functioning as a sentiment evaluator, identify if the given sentence is 'positive' or 'negative'. Categorize: "
        ],
        'qqp': [
            "Are these two questions semantically the same? Please answer with either 'equivalent' or 'not_equivalent'.",
            "Determine if these two questions are equivalent. Respond with 'equivalent' or 'not_equivalent'.",
            "Determine if the provided pair of statements can be regarded as the same in meaning by responding with 'equivalent' or 'not_equivalent'.",
            "In your capacity as a question comparison tool, evaluate the following pair of questions and categorize them as 'equivalent' or 'not_equivalent'.",
            "In the role of a question equivalence detection system, examine the given questions and reply with 'equivalent' if they share the same meaning or 'not_equivalent' if they differ.",
            "Operating as a question similarity assessment tool, analyze the provided questions and determine whether they convey the same meaning, responding with 'equivalent' or 'not_equivalent'.",
        ],
        'mnli': [
            "Evaluate the link between the sentences and categorize it as either 'entailment', 'neutral', or 'contradiction':",
            "Identify if the correlation between the sentences is 'entailment', 'neutral', or 'contradiction'. Reply with 'entailment', 'neutral', or 'contradiction':",
            "Review the two sentences and decide if they show entailment, neutrality, or contradiction. Respond by choosing one of the following: 'entailment', 'neutral', or 'contradiction'.",
            "As a tool for analyzing entailment, scrutinize the relationship in the sentences and designate it as 'entailment', 'neutral', or 'contradiction':",
            "In your capacity as an entailment detection system, scrutinize the linkage between the sentences and indicate 'entailment', 'neutral', or 'contradiction':",
            "Operating as an entailment assessment tool, examine the sentences given and determine their relation as 'entailment', 'neutral', or 'contradiction':",
        ],
        'qnli': [
            "Review the context and the posed question, then decide whether the answer logically follows by choosing 'entailment' or 'not_entailment'.",
            "Assess whether the answer is a logical conclusion from the given context and question by selecting 'entailment' or 'not_entailment'.",
            "Analyze the provided context and question to determine if the answer is supported, responding with 'entailment' or 'not_entailment'.",
            "As an expert in linguistics, evaluate if the answer is entailed by the given context in response to the question, with options 'entailment' or 'not_entailment'.",
            "In the capacity of a semantic analyst, decide if the context validates the answer to the posed question, choosing either 'entailment' or 'not_entailment'.",
            "As someone analyzing text, scrutinize whether the context implies the answer to the question, responding with either 'entailment' or 'not_entailment'."
        ],
        'rte': [
            "Ascertain whether the presented sentence pair exemplifies 'entailment' or 'not_entailment'. Your response should be either 'entailment' or 'not_entailment'.",
            "For the given sentence duo, is it 'entailment' or 'not_entailment'? Please reply with only 'entailment' or 'not_entailment'.",
            "Evaluate the connection between these sentences as 'entailment' or 'not_entailment'. Please respond with either 'entailment' or 'not_entailment'.",
            "Acting as an entailment analysis instrument, determine if the relationship between these sentences is 'entailment' or 'not_entailment'.",
            "In the capacity of an entailment detection system, scrutinize the link between these sentences and indicate 'entailment' or 'not_entailment'.",
            "Operating as an entailment assessment tool, scrutinize the given sentences and conclude whether they exhibit 'entailment' or 'not_entailment'."
        ],
        'cola': [
            "Evaluate the sentence below and determine if its grammar is 'Acceptable' or 'Unacceptable':",
            "Review the sentence that follows and decide if its grammar is correct. Indicate 'Acceptable' or 'Unacceptable':",
            "Scrutinize the sentence provided and judge its grammatical accuracy. Reply with 'Acceptable' or 'Unacceptable':",
            "Acting as a grammar analysis tool, evaluate the sentence below and denote it as 'acceptable' if grammatically correct, or 'unacceptable' if not:",
            "As a system for grammar verification, scrutinize the sentence given and mark as 'acceptable' for grammatical correctness or 'unacceptable' for errors:",
            "Working as a tool for grammar assessment, inspect the presented sentence and determine its grammatical validity, indicating 'acceptable' or 'unacceptable':"
        ],
        'mrpc': [
            "Is the semantic content of these sentences identical? Please answer with 'equivalent' or 'not_equivalent'.",
            "Do these sentences convey the same meaning? Please respond with 'equivalent' or 'not_equivalent'.",
            "Are these two sentences semantically the same? Respond with 'equivalent' or 'not_equivalent'.",
            "As an expert in semantic analysis, determine whether these sentences are 'equivalent' or 'not_equivalent'.",
            "In your role as a linguistic analyst, evaluate these sentences and categorize their similarity as 'equivalent' or 'not_equivalent'.",
            "As an evaluator of sentence similarity, examine these sentences and determine if they are 'equivalent' or 'not_equivalent'."
        ],
        'wnli': [
            "Determine if the pair of sentences shown are an example of entailment or not_entailment. Reply with either 'entailment' or 'not_entailment'.",
            "Is the relationship between these two sentences entailment or not_entailment? Provide a one-word response, either 'entailment' or 'not_entailment'.",
            "Evaluate the given sentences and indicate if they represent entailment or not_entailment. Please respond with 'entailment' or 'not_entailment'.",
            "In the capacity of an entailment analysis tool, classify the relationship of the provided sentences as 'entailment' or 'not_entailment'.",
            "Operating as an entailment detection system, scrutinize the link between these sentences and reply with 'entailment' or 'not_entailment'.",
            "As an entailment assessment tool, review the given sentences and determine their relationship as either 'entailment' or 'not_entailment'."
        ]
    }
}

LANGUAGES = {
    'ar': 'Arabic',
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'ru': 'Russian',
    'zh': 'Chinese',
    'it': 'Italian',
    'nl': 'Dutch',
    'ro': 'Romanian',
    'ja': 'Japanese',
    'ko': 'Korean',
}

MATH_QUESTION_TYPES = {
    'algebra_linear_1d': ' linear algebra ',
    'algebra_linear_2d': ' linear algebra ',
    'algebra_sequence_next_term': ' given a sequence predict the next term ',
    'arithmetic_addition_sub_multiple': ' arithmetic addition and subtraction ',
    'arithmetic_mul_div_multiple': ' arithmetic multiplication and division ',
    'arithmetic_mixed': ' arithmetic addition, subtraction, multiplication and division ',
    'arithmetic_nearest_integer_root': ' arithmetic nearest integer root ',
    'comparison_closest': ' compare which one of given numbers is closest to target number ',
    'comparison_kth_biggest': ' compare which one of given numbers is kth biggest or smallest ',
    'comparison_pair': ' comparison which one of given numbers is bigger or smaller ',
    'measurement_conversion': ' measurement conversion ',
    'numbers_base_conversion': ' numbers base conversion ',
    'numbers_div_remainder': ' numbers division and remainder ',
    'numbers_gcd': ' numbers greatest common divisor ',
    'numbers_is_factor': ' if one number is a factor of antoher number ',
    'number_is_prime': ' if a number is prime ',
    'numbers_lcm': ' least common multiple ',
    'numbers_place_value': ' place value ',
    'numbers_round_number': ' round number ',
    'polynomials_evaluate': ' polynomials evaluate ',
}

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

with open('evidence.json', 'r', encoding='utf-8') as f:
    evi = f.read()
evidence_json = json.loads(evi)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = SentenceTransformer('bert-base-nli-mean-tokens')
model1 = SentenceTransformer("paraphrase-MiniLM-L3-v2")
model.to(device)
model1.to(device)
model.eval()
model1.eval()


def similar_cal(sen_list1, sen_list2, mapping=None, threshold=0.7):
    if mapping is None:
        mapping = {0: [i for i in range(len(sen_list2))]}
    embeddings1 = model.encode(sen_list1, convert_to_tensor=True)
    embeddings2 = model.encode(sen_list2, convert_to_tensor=True)
    cosine_scores1 = util.cos_sim(embeddings1, embeddings2)
    embeddings3 = model1.encode(sen_list1, convert_to_tensor=True)
    embeddings4 = model1.encode(sen_list2, convert_to_tensor=True)
    cosine_scores2 = util.cos_sim(embeddings3, embeddings4)
    cosine_scores = np.array(cosine_scores1.cpu() * .5 + cosine_scores2.cpu() * .5)

    tmp = [j for j, _ in
           np.argwhere(
               np.array(cosine_scores) > threshold)]  # which sentence has score higher than threshold, give sentence
    # index
    length = cosine_scores.shape[0]  # how many sentences
    sentence_index = [i for n, i in enumerate(tmp) if i not in tmp[:n]]  # drop duplicate
    index = [int(np.argmax(cosine_scores, axis=1)[i]) for i in
             sentence_index]  # sentence relate to which target sentence, give sentence index sentence index to
    # category index
    sentence_label = [
        255 if i not in sentence_index else [j for j in mapping if index[sentence_index.index(i)] in mapping[j]][0] for
        i in range(
            length)]  # Give a list which show each sentence category. If one sentence has no category, it will show
    # 255. otherwise will show index of category. EX: [255,255,1,255,0]
    return index, sentence_index, sentence_label


def map_value(relevant_fact):  # mapping sentence in json with each category(target sentence)
    y = relevant_fact['relevant_facts']
    support = [k for i in y for k in i['sentences']]  # all target sentences support yes or no
    support_fact = [i['describe'] for i in y]  # all categories support yes or no
    support_l = [i['sentences'] for i in y]
    mapping = {v: [support.index(i) for i in k] for v, k in enumerate(support_l)}  # {0:[0,1],1:[2,3],2:[4,5,6]}

    return support, support_fact, mapping


#  match sentence index to category index
def match(value, mapping, support_fact):
    fact_lack = support_fact * 1
    fact_present = []
    for item in value:
        index = [i for i in mapping if item in mapping[i]][0]
        if support_fact[index] not in fact_present:
            fact_present.append(support_fact[index])
            fact_lack.remove(support_fact[index])

    return fact_present, fact_lack


def cal(lesson_id, sentences):
    lesson_type = evidence_json[lesson_id]['prompt_type']
    if lesson_type == 1:

        relevant_fact_y = evidence_json[lesson_id]['support_yes']
        support_y, support_fact_y, mapping_y = map_value(relevant_fact_y)
        value_y, sentence_index_y, sentences_label_y = similar_cal(sentences, support_y, mapping_y)
        fact_present_y, fact_lack_y = match(value_y, mapping_y, support_fact_y)

        relevant_fact_n = evidence_json[lesson_id]['support_no']
        support_n, support_fact_n, mapping_n = map_value(relevant_fact_n)
        value_n, sentence_index_n, sentences_label_n = similar_cal(sentences, support_n, mapping_n)
        fact_present_n, fact_lack_n = match(value_n, mapping_n, support_fact_n)
        result = {'num_support_y': len(sentence_index_y), 'fact_present_y': fact_present_y,
                  'fact_lack_y': fact_lack_y, 'fact_all_y': support_fact_y, 'sentence_label_y': sentences_label_y,
                  'num_support_n': len(sentence_index_n), 'facts_present_n': fact_present_n,
                  'fact_lack_n': fact_lack_n, 'fact_all_n': support_fact_n, 'sentence_label_n': sentences_label_n,
                  'number_general_info': 0, 'sentences_label_ge': [255] * len(sentences), 'number_solution': 0,
                  'sentences_label_sol': [255] * len(sentences)}
        general_info = evidence_json[lesson_id]['general_info']
        solution = evidence_json[lesson_id]['solution']
        if general_info:
            _, sentence_index_ge, sentences_label_ge = similar_cal(sentences, general_info)
            result['number_general_info'] = len(sentence_index_ge)
            result['sentences_label_ge'] = sentences_label_ge
        if solution:
            _, sentence_index_sol, sentences_label_sol = similar_cal(sentences, solution)
            result['number_solution'] = len(sentence_index_sol)
            result['sentences_label_sol'] = sentences_label_sol

        return result
    else:
        return {'general_info': 0}


app = Flask(__name__)  ####
CORS(app)


@app.route("/predict", methods=['GET', 'POST'])
def run():
    if request.method == 'POST':  ####POST
        # data_fz = json.loads(request.get_data().decode('utf-8')) ####get data
        data_fz = request.get_json()
        # print(data_fz)

        if data_fz is not None:
            # data_fz = request.to_dict()
            lesson_id = data_fz['lesson_id']
            content = data_fz['content']

        else:
            return jsonify({'Bj': -1, 'Mess': '', 'type': 'Error'})  ####return -1 if no data
    else:
        return jsonify({'Bj': -2, 'Mess': '', 'type': 'Error'})  #### return -2 if not right format

    label = cal(lesson_id, content)
    # print(label)

    return jsonify(label)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8008, debug=False)

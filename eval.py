# Copyright 2023 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


import time
import pandas as pd
import jsonlines
import openai
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt_dic = {"system_prompt": "You are a helpful and precise assistant for checking the quality of the answer.",
              "prompt_template": "[Detailed Image Description]\n{human_annotation}\n\n[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
              "defaults": {
                  "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question and image description displayed above. AI assistants are provided with detailed image description and questions.\nPlease rate the helpfulness, relevance, accuracy, comprehensiveness of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."},
              "description": "Prompt for general questions", "category": "general"}

MAX_API_RETRY = 5

def parse_args():
    parser = argparse.ArgumentParser(description="TouchStone evaluation.")
    parser.add_argument("submit_file", help="submitted tsv file")
    parser.add_argument("openai_key", help="openai api key")
    parser.add_argument("-n","--model-name",default="Assessed Model.", help="Method name")
    args = parser.parse_args()
    return args


def process_reply(reply):
    # Process the response to the scores
    generated, reason = reply.strip().split('\n', 1)
    score1, score2 = generated.strip().split(' ')
    score1, score2 = float(score1), float(score2)
    return score1, score2, reason


def evaluate(query):
    """
    Evaluate the response using the GPT-4 model.
    Args:
    - query: str, the user's query

    Returns:
    - result: tuple(int, int, str), the evaluation result in the format (score1, score2, response)
      - score1: int, the first score
      - score2: int, the second score
      - response: str, the status from the model
    """

    logging.basicConfig(level=logging.INFO)
    # Retry the API call for a maximum of MAX_API_RETRY times
    for i in range(MAX_API_RETRY):
        try:
            # Make API call to the Chat Completion API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt_dic['system_prompt']},
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                stop=["\n"],
                temperature=0, 
            )
            
            content = response["choices"][0]["message"]["content"]
            logger.info(content)

            # Process the response
            reply = process_reply(content+'\nok')

            if len(reply) == 3:
                return reply[0], reply[1], reply[2]
            else:
                return 0, 0, "error"
        
        except Exception as e:
            logger.error(e)
            time.sleep(5)

    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return 0, 0, "error"


def process(file, name):
    # Load data from file
    print(' -------------- evaluate {} ------------- '.format(name))
    df = pd.read_csv(file, sep='\t')
    num_answers = len(df)
    column_names = df.columns.tolist()
    results = []
    
    # Check if the required columns are present in the file
    assert num_questions == 908, "Insufficient number of answers provided."
    assert "response" in column_names, "The responses of model must be in the submit file."
    assert "question" in column_names, "The questions must be in the submit file."
    assert "human_annotation" in column_names, "The human_annotation must be in the submit file."
    assert "gpt4_ha_answer" in column_names, "The gpt4_ha_answer must be in the submit file."
    assert "category" in column_names, "The category must be in the submit file."
    assert "task_name" in column_names, "The task_name must be in the submit file."
    
    for idx in range(num_answers):
        # Process each answer
        ques = df.iloc[idx]['question']
        human_annotation = df.iloc[idx]['human_annotation']
        ans1 = df.iloc[idx]['gpt4_ha_answer']
        ans2 = df.iloc[idx]['response']
        
        # Check if ans2 is a list and convert it to a string
        if isinstance(ans2, list):
            ans2 = ans2[0]
            
        # Create prompt using the provided template
        prompt = prompt_dic["defaults"]["prompt"]
        query1 = prompt_dic["prompt_template"].format(human_annotation=human_annotation, question=ques, answer_1=ans1,
                                                      answer_2=ans2, prompt=prompt)
        query2 = prompt_dic["prompt_template"].format(human_annotation=human_annotation, question=ques, answer_1=ans2,
                                                      answer_2=ans1, prompt=prompt)
        
        # Evaluate the answer with position balancing
        round1_score1, round1_score2, round1_reason = evaluate(query1)
        round2_score1, round2_score2, round2_reason = evaluate(query2)
        
        # Create a dictionary to store the results
        result_dic = {
            'ques_id': idx,
            'human_annotation': df.iloc[idx]['human_annotation'],
            'category': df.iloc[idx]['category'],
            'task_name': df.iloc[idx]['task_name'],
            'round1': {
                'model1': 'gpt4-ha',
                'model2': name,
                'answer1': ans1,
                'answer2': ans2,
                'score1': round1_score1,
                'score2': round1_score2,
                'reason': round1_reason
            },
            'round2': {
                'model1': name,
                'model2': 'gpt4-ha',
                'answer1': ans2,
                'answer2': ans1,
                'score1': round2_score1,
                'score2': round2_score2,
                'reason': round2_reason
            }
        }
        
        results.append(result_dic)
    
    # Write results to a JSONL file
    with jsonlines.open('evaluation_results_{}.jsonl'.format(name), 'w') as w:
        for result_dic in results:
            w.write(result_dic)
    
    return 'evaluation_results_{}.jsonl'.format(name)

def compute_score(review_file):
    # Dictionary to store scores for each model
    scores = {}
    # Dictionary to store scores categorized by category and model
    scores_cate_wise = {}

    # Open the JSONL file
    with jsonlines.open(review_file, "r") as jsonreader:
        # Read each line of the file
        for i, info in enumerate(jsonreader):
            # Get round1 and round2 information from each line
            round1 = info["round1"]
            round2 = info["round2"]
            rounds = [round1, round2]

            # Iterate over both rounds
            for round_ in rounds:
                # Check if the reason is not 'error'
                if round_['reason'] != 'error':
                    model1 = round_["model1"]
                    model2 = round_["model2"]

                    # Check if the model1 is not in scores dictionary
                    if model1 not in scores:
                        scores[model1] = []
                    # Check if the model2 is not in scores dictionary
                    if model2 not in scores:
                        scores[model2] = []

                    # Append the scores for model1 and model2
                    scores[model1].append(round_["score1"])
                    scores[model2].append(round_["score2"])

                    # Split the category string and iterate over each category
                    for cate in info["category"].split(','):
                        # Check if the category is not in scores_cate_wise dictionary
                        if cate not in scores_cate_wise:
                            scores_cate_wise[cate] = {}
                        # Check if the model1 is not in scores_cate_wise dictionary for the category
                        if model1 not in scores_cate_wise[cate]:
                            scores_cate_wise[cate][model1] = []
                        # Check if the model2 is not in scores_cate_wise dictionary for the category
                        if model2 not in scores_cate_wise[cate]:
                            scores_cate_wise[cate][model2] = []

                        # Append the scores for model1 and model2 categorized by category
                        scores_cate_wise[cate][model1].append(round_["score1"])
                        scores_cate_wise[cate][model2].append(round_["score2"])

    print(' -------------- TouchStone Overall Score ------------- ')
    # Calculate and print the mean scores for each model
    for model_name in scores:
        scores[model_name] = np.mean(scores[model_name]) * 100
        print(model_name, scores[model_name])

    # Print the scores categorized by category
    for cate in scores_cate_wise:
        print(' -------------- {} ------------- '.format(cate))
        for model_name in scores_cate_wise[cate]:
            # Calculate and print the mean scores for each model in each category
            print(model_name, np.mean(scores_cate_wise[cate][model_name]) * 100)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    openai.api_key = args.openai_key
    review_file = process(args.submit_file, args.model_name)
    compute_score(review_file)

   



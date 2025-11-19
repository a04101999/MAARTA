import time
import json
import argparse
import random
from model_request.req import togetherai_request, extract_json, validate_prediction
import os
from dotenv import load_dotenv

# ---------------------------
# Data loading (Manually adjust if not using a .env file)
# ---------------------------

load_dotenv()
api_key = os.getenv("TOGETHERAI_API_KEY")
assert api_key is not None and len(
    api_key) > 0, "Please set the TOGETHERAI_API_KEY environment variable."

metadata_file = os.getenv("METADATA_FILE")
assert api_key is not None and len(
    api_key) > 0, "Please set the METADATA_FILE environment variable."

data_file = os.getenv("DATA_FILE")
assert api_key is not None and len(
    api_key) > 0, "Please set the DATA_FILE environment variable."

metadata_file_path = f'../{metadata_file}'
data_file_path = f'../{data_file}'

with open(metadata_file_path, 'r') as file:
    datalab = json.load(file)

with open(data_file_path, 'r') as file:
    data = json.load(file)

# ---------------------------
# Single Agent Inference Functions
# ---------------------------


def create_zero_shot_prompt(experienced_data, inexperienced_data, experienced_time_stamps, inexperienced_time_stamps):
    prompt = f"""
    ### Experienced Radiologist:
    - Findings: {experienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(experienced_time_stamps, indent=2)}
    - Eye Gaze Data:
        - Fixations: {json.dumps(experienced_data['fixations'], indent=2)}
        - Durations: {json.dumps(experienced_data['durations'], indent=2)}

    ### Inexperienced Radiologist:
    - Findings: {inexperienced_data['transcript']}
    - Time-Stamped Text: {json.dumps(inexperienced_time_stamps, indent=2)}
    - Eye Gaze Data:
        - Fixations: {json.dumps(inexperienced_data['fixations'], indent=2)}
        - Durations: {json.dumps(inexperienced_data['durations'], indent=2)}

    ### Task:
    Compare the findings, time-stamped text, and eye gaze data on the same medical imaging of both radiologists to address the following:

    1. Identify missed findings by the inexperienced radiologist.
    2. Analyze fixation and duration patterns to determine differences in focus and attention.

    Explain reasoning before returning the response.

    ### Response:
    Always provide an answer in JSON format, Answer: 1 for Yes and 0 for No:
    {{
        "Missed abnormality due to missing fixation": ,
        "Missed abnormality due to reduced fixation": ,
        "Missed abnormality due to incomplete knowledge": ,
        "No missing abnormality":
    }}
    """

    return prompt


def run_inference(data: dict, saved_predictions_to_JSON: dict):
    system_role_prompt = "You are a helpful teaching assistant to provide feedback to the inexperienced radiologist."
    number_processed = 0

    class_labels = set({'Missed abnormality due to missing fixation', 'Missed abnormality due to reduced fixation',
                       'Missed abnormality due to incomplete knowledge', 'No missing abnormality'})

    for K in data.items():
        if K[0] in saved_predictions_to_JSON:
            continue

        pred_start_time = time.time()
        da = data[K[0]]['correct_data']

        s = ''
        exp_tran = da['transcript']
        for i in exp_tran:
            s = s + i['sentence']
        fixations_dur = da['FPOGD']

        fixations_exp = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

        da = data[K[0]]['incorrect_data']
        if len(da) == 0:
            da = data[K[0]]['correct_data']
            si = ''
            inexp_tran = da['transcript']
            for i in inexp_tran:
                si = si + i['sentence']
            fake_durations = da['FPOGD']
            fake_fixations = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

        else:
            si = ''
            inexp_tran = da['transcript']
            for i in inexp_tran:
                si = si + i['sentence']
            fake_durations = da['FPOGD']
            fake_fixations = list(zip(da['X_ORIGINAL'], da['Y_ORIGINAL']))

        # Prepare the data
        experienced_data = {
            'transcript': s,
            'fixations': fixations_exp,
            'durations': fixations_dur
        }

        inexperienced_data = {
            'transcript': si,
            'fixations': fake_fixations,
            'durations': fake_durations
        }

        prompt = create_zero_shot_prompt(
            experienced_data, inexperienced_data, exp_tran, inexp_tran)

        response = togetherai_request(system_role_prompt, prompt, api_key,
                                      model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", temperature=0.2, max_tokens=500)

        if response is None:
            return True

        error_assessment = extract_json(response)

        if validate_prediction(error_assessment, class_labels):
            pred_end_time = time.time()
            saved_predictions_to_JSON[K[0]] = error_assessment
            saved_predictions_to_JSON[K[0]
                                      ]['inference_time'] = pred_end_time - pred_start_time
            print(K[0], "processed in", pred_end_time -
                  pred_start_time, "seconds.")
            number_processed += 1

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Your script description here.")
    parser.add_argument('--results', type=str,
                        help='Path to preexisting results file', required=False)
    args = parser.parse_args()

    random_sampled_data = {key: data[key]
                           for key in random.sample(list(data.keys()), len(data))}
    assert len(random_sampled_data) == len(data)
    dataset_size = len(random_sampled_data)

    batch_size = int(dataset_size / 20)
    batches = [{} for _ in range(20)]

    for i, (key, value) in enumerate(random_sampled_data.items()):
        batch_index = i // batch_size
        if batch_index >= 20:
            batch_index = 19
        batches[batch_index][key] = value

    print("Batch Size", batch_size, "| Last Batch Size", len(batches[-1]))
    print("Total Number of Samples", sum(len(batch) for batch in batches))
    print("===============================================\n")

    assert sum(len(batch) for batch in batches) == dataset_size

    saved_predictions_to_JSON = {}
    results_output_file = args.results if args.results else "llama_3.2_11B_single_agent_results.json"

    if args.results:
        print("Loading existing results from", args.results)
        with open(args.results, 'r') as file:
            saved_predictions_to_JSON = json.load(file)

        print("Loaded", len(saved_predictions_to_JSON),
              "existing predictions.")
        print("===============================================\n")

    current_batch = 1

    for b in batches:
        print("Inference on batch", current_batch)
        is_error = run_inference(
            b, saved_predictions_to_JSON)

        print("Completed batch", current_batch)
        with open(results_output_file, 'w') as file:
            json.dump(saved_predictions_to_JSON, file, indent=4)
        current_batch += 1

        if is_error:
            print("============================================================")
            print("TogetherAI request issue, please check if API key has enough credits.")
            print("============================================================")
            return

        print("===============================================\n")

    with open(results_output_file, 'w') as file:
        json.dump(saved_predictions_to_JSON, file, indent=4)

    # Perform a final check to see if all data has been processed
    missed_samples = {}

    for key in random_sampled_data.keys():
        if key not in saved_predictions_to_JSON.keys():
            missed_samples[key] = random_sampled_data[key]

    print("Missed samples:", len(missed_samples))
    print("===============================================\n")

    missed_batches = []

    for i in range(0, len(missed_samples), batch_size):
        missed_batches.append({k: missed_samples[k] for k in list(
            missed_samples.keys())[i:i + batch_size]})

    epochs = 0

    while len(saved_predictions_to_JSON) < len(random_sampled_data) and epochs < 5:
        for i, b in enumerate(missed_batches):
            is_error = run_inference(
                b, saved_predictions_to_JSON)

            print(f"Completed batch {i + 1}")
            with open(results_output_file, 'w') as file:
                json.dump(saved_predictions_to_JSON, file, indent=4)

            if is_error:
                print("============================================================")
                print(
                    "TogetherAI request issue, please check if API key has enough credits.")
                print("============================================================")
                return

            print("===============================================\n")

        epochs += 1

    with open(results_output_file, 'w') as file:
        json.dump(saved_predictions_to_JSON, file, indent=4)

    print("DONE")


if __name__ == "__main__":
    main()

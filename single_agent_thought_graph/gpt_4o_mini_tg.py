import time
import json
import argparse
import random
from dotenv import load_dotenv
import os
from model_request.req import openai_request, extract_json, validate_prediction
from thought_graph.graph_creation import create_scene_graph_by_sentence, compare_scene_graphs_with_llm


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None and len(
    api_key) > 0, "Please set the OPENAI_API_KEY environment variable."


def run_thought_graph_inference(data, saved_thought_graph_results_toJSON: dict):
    system_role_prompt = "You are a helpful radiology teaching assistant to provide feedback to the inexperienced radiologist."
    number_processed = 0
    class_labels = set({'No Missing Subgraph', 'Missing Subgraph due to Missing fixation',
                       'Missing Subgraph due to reduced fixation duration', 'Missing Subgraph due to undefined reason'})

    for K in data.items():
        if K[0] in saved_thought_graph_results_toJSON:
            continue

        pred_start_time = time.time()

        da = data[K[0]]['correct_data']
        result = [
            {
                'X_ORIGINAL': da['X_ORIGINAL'][i],
                'Y_ORIGINAL': da['Y_ORIGINAL'][i],
                'FPOGD': da['FPOGD'][i],
                'Time (in secs)': da['Time (in secs)'][i]
            }
            for i in range(len(da['X_ORIGINAL']))
        ]
        # exp_fix = result.copy()
        timestamped_report = da['transcript']
        fixation_data = result

        # Create the scene graph for experienced radiologist
        scene_graph_output = create_scene_graph_by_sentence(
            timestamped_report, fixation_data)

        if len(data[K[0]]['incorrect_data']) == 0:

            da = data[K[0]]['correct_data']
        else:
            da = data[K[0]]['incorrect_data']
        result = [
            {
                'X_ORIGINAL': da['X_ORIGINAL'][i],
                'Y_ORIGINAL': da['Y_ORIGINAL'][i],
                'FPOGD': da['FPOGD'][i],
                'Time (in secs)': da['Time (in secs)'][i]
            }
            for i in range(len(da['X_ORIGINAL']))
        ]
        # inexp_fix = result.copy()
        timestamped_reporti = da['transcript']
        fixation_datai = result

        # Create the scene graph for inexperienced radiologist
        scene_graph_outputi = create_scene_graph_by_sentence(
            timestamped_reporti, fixation_datai)

        scene_graph_exp = scene_graph_output
        scene_graph_inexp = scene_graph_outputi

        prompt = compare_scene_graphs_with_llm(
            scene_graph_exp, scene_graph_inexp)

        response = openai_request(system_role_prompt, prompt, api_key,
                                  model="gpt-4o-mini", temperature=0.2, max_tokens=500)

        if response is None:
            continue

        error_assessment = extract_json(response)

        if error_assessment is None:
            continue

        if validate_prediction(error_assessment, class_labels):
            pred_end_time = time.time()
            saved_thought_graph_results_toJSON[K[0]] = error_assessment
            saved_thought_graph_results_toJSON[K[0]
                                               ]['inference_time'] = pred_end_time - pred_start_time
            print(K[0], "processed in", pred_end_time -
                  pred_start_time, "seconds.")
            number_processed += 1


def main():
    metadata_file = os.getenv("METADATA_FILE")
    assert api_key is not None and len(
        api_key) > 0, "Please set the METADATA_FILE environment variable."

    data_file = os.getenv("DATA_FILE")
    assert api_key is not None and len(
        api_key) > 0, "Please set the DATA_FILE environment variable."

    parser = argparse.ArgumentParser(
        description="Your script description here.")
    parser.add_argument('--results', type=str,
                        help='Path to preexisting results file', required=False)
    args = parser.parse_args()

    with open(f'../{metadata_file}', 'r') as file:
        datalab = json.load(file)

    with open(f'../{data_file}', 'r') as file:
        data = json.load(file)

    random_sampled_data = {key: data[key]
                           for key in random.sample(list(data.keys()), len(data))}
    assert len(random_sampled_data) == len(data)
    dataset_size = len(random_sampled_data)

    batch_size = int(dataset_size / 20)
    batches = [{} for _ in range(20)]

    random_id = random.randint(0, dataset_size - 1)
    ex_dicom_id = list(random_sampled_data.keys())[random_id]

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
    results_output_file = args.results if args.results else "gpt4o_mini_tg_results.json"

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
        run_thought_graph_inference(
            b, saved_predictions_to_JSON)

        print("Completed batch", current_batch)
        with open(results_output_file, 'w') as file:
            json.dump(saved_predictions_to_JSON, file, indent=4)
        current_batch += 1

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
            run_thought_graph_inference(
                b, saved_predictions_to_JSON)

            print(f"Completed batch {i + 1}")
            with open(results_output_file, 'w') as file:
                json.dump(saved_predictions_to_JSON, file, indent=4)
            print("===============================================\n")

        epochs += 1

    with open(results_output_file, 'w') as file:
        json.dump(saved_predictions_to_JSON, file, indent=4)

    print("DONE")


if __name__ == "__main__":
    main()

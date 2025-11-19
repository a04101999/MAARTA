import os
import numpy as np
import json
import re
import openai
import random
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import time
from dotenv import load_dotenv
import os
import argparse

# ---------------------------
# OpenAI API key configuration
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None and len(
    api_key) > 0, "Please set the OPENAI_API_KEY environment variable."

openai.api_key = api_key

# ---------------------------
# Data loading
# ---------------------------
metadata_file = os.getenv("METADATA_FILENAME")
assert api_key is not None and len(
    api_key) > 0, "Please set the METADATA_FILENAME environment variable."

data_file = os.getenv("DATA_FILENAME")
assert api_key is not None and len(
    api_key) > 0, "Please set the DATA_FILENAME environment variable."

with open(metadata_file, 'r') as file:
    datalab = json.load(file)

with open(data_file, 'r') as file:
    data = json.load(file)

parser = argparse.ArgumentParser(
    description="Add a prior result JSON file to continue an existing experiment.")
parser.add_argument('--results', type=str,
                    help='Path to preexisting results file', required=False)
args = parser.parse_args()

result_output_file = args.results if args.results else "missed_findings_results.json"


# Shuffle data
data = {key: data[key]
        for key in random.sample(list(data.keys()), len(data))}


# ---------------------------
# Helper functions for scene graph
# ---------------------------


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_closest_times(fixation_data, begin_time, end_time):
    closest_begin_time = max([fix['Time (in secs)']
                             for fix in fixation_data if fix['Time (in secs)'] <= begin_time], default=None)
    closest_end_time = min([fix['Time (in secs)']
                           for fix in fixation_data if fix['Time (in secs)'] >= end_time], default=None)
    return closest_begin_time, closest_end_time


def create_subgraph(sentence_text, fixation_nodes, sentence_counter):
    subgraph = {
        "Abnormality": sentence_text,
        "nodes": fixation_nodes,
        "edges": []
    }

    position_tracker = {}
    for i, current_node in enumerate(fixation_nodes):
        position_key = tuple(current_node["fixation_point"])
        if position_key in position_tracker:
            original_node = position_tracker[position_key]
            subgraph["edges"].append({
                "from": original_node["id"],
                "to": current_node["id"],
                "type": "revisit"
            })
        else:
            position_tracker[position_key] = current_node

        if i > 0:
            previous_node = fixation_nodes[i - 1]
            dist = euclidean_distance(
                previous_node["fixation_point"], current_node["fixation_point"])
            subgraph["edges"].append({
                "from": previous_node["id"],
                "to": current_node["id"],
                "distance": round(dist, 3)
            })

    return subgraph


def create_scene_graph_by_sentence(timestamped_report, fixation_data):
    scene_graph = {"scene_graph": {"subgraphs": [], "inter_phrase_edges": []}}
    sentence_counter = 1
    phrase_time_ranges = []
    idle_fixations = []

    for sentence_info in timestamped_report:
        sentence_text = sentence_info['sentence']
        begin_time, end_time = sentence_info['begin_time'], sentence_info['end_time']
        phrase_time_ranges.append((begin_time, end_time))
        closest_begin_time, closest_end_time = find_closest_times(
            fixation_data, begin_time, end_time)
        if closest_begin_time is None or closest_end_time is None:
            continue

        fixation_nodes = []
        for fixation in fixation_data:
            fixation_time = fixation['Time (in secs)']
            if closest_begin_time <= fixation_time <= closest_end_time:
                fpx, fpy, fpd = fixation['FPOGX'], fixation['FPOGY'], fixation['FPOGD']
                node_id = f"fixation_{sentence_counter * 10 + len(fixation_nodes)}"
                fixation_nodes.append({
                    "id": node_id,
                    "fixation_point": [fpx, fpy],
                    "fixation_duration": fpd
                })

        subgraph = create_subgraph(
            sentence_text, fixation_nodes, sentence_counter)
        scene_graph["scene_graph"]["subgraphs"].append(subgraph)

        if len(scene_graph["scene_graph"]["subgraphs"]) > 1:
            previous_phrase = scene_graph["scene_graph"]["subgraphs"][-2]["Abnormality"]
            scene_graph["scene_graph"]["inter_phrase_edges"].append({
                "from": previous_phrase,
                "to": sentence_text,
                "relationship": "sequence"
            })

        sentence_counter += 1

    for fixation in fixation_data:
        fixation_time = fixation['Time (in secs)']
        if not any(begin <= fixation_time <= end for begin, end in phrase_time_ranges):
            node_id = f"idle_fixation_{len(idle_fixations)}"
            idle_fixations.append({
                "id": node_id,
                "fixation_point": [fixation['FPOGX'], fixation['FPOGY']],
                "fixation_duration": fixation['FPOGD']
            })

    idle_subgraph = create_subgraph("Idle Subgraph", idle_fixations, 0)
    scene_graph["scene_graph"]["subgraphs"].append(idle_subgraph)

    return scene_graph


# ---------------------------
# MAARTA setup: Principal LLM and User Proxy
# ---------------------------
principal_llm = AssistantAgent(
    name="PrincipalLLM",
    description="Analyzes radiology reports to detect missed abnormalities and incorrect decisions.",
    llm_config={"model": "gpt-4o-mini", "api_key": openai.api_key}
)

user_proxy = UserProxyAgent(
    name="UserProxy",
    description="Interacts with the Principal LLM to request analysis.",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}
)

group_chat = GroupChat(agents=[principal_llm, user_proxy], messages=[])
manager = GroupChatManager(
    name="RadiologyComparisonManager", groupchat=group_chat)

# ---------------------------
# Helper functions for MAARTA
# ---------------------------


def recruit_agents_for_comparison(subgraphs, scene_graph_inexp):
    """
    Creates and manages comparison agents to analyze subgraphs of missed abnormalities.
    For each subgraph A in subgraphs, recruits agents to compare A with each subgraph in
    scene_graph_inexp['scene_graph']['subgraphs']. Combines the results for each A using
    logical OR and returns the final JSON response.
    """

    if not subgraphs:
        print("No subgraphs provided. Returning None.")
        return None

    if "scene_graph" not in scene_graph_inexp or "subgraphs" not in scene_graph_inexp["scene_graph"]:
        print("No subgraphs found in the inexperienced radiologist's scene graph. Returning None.")
        return None

    inexp_subgraphs = scene_graph_inexp["scene_graph"]["subgraphs"]

    combined_results = {}

    for i, subgraph_A in enumerate(subgraphs):
        abnormality_A = subgraph_A["Abnormality"]
        print(f"Processing subgraph A: {abnormality_A}")

        # Create a mapping of agents for this subgraph A
        comparison_agents = {
            f"ComparisonAgent_{i}_{j}": AssistantAgent(
                name=f"ComparisonAgent_{i}_{j}",
                description=f"Compares subgraph A ({abnormality_A}) with subgraph {j} from the inexperienced radiologist.",
                llm_config={
                    "model": "gpt-4o-mini",
                    "api_key": openai.api_key
                }
            )
            for j in range(len(inexp_subgraphs))
        }

        # Register agents with the manager
        manager.groupchat.agents.extend(comparison_agents.values())

        results_for_A = []

        # Perform comparisons for subgraph A
        for j, subgraph_inexp in enumerate(inexp_subgraphs):
            agent = comparison_agents.get(f"ComparisonAgent_{i}_{j}")

            if not agent:
                print(f"Agent not found for comparison {i}_{j}, skipping.")
                continue

            # Perform analysis
            response = user_proxy.initiate_chat(
                agent,
                message=(
                    f"Let's perform the analysis step by step:\n\n"
                    "1. **Extract Fixation Points**:\n"
                    "   - Extract all fixation coordinates (e.g., [x, y]) from the extracted subgraph A ({abnormality_A}) and the subgraph {j} from the inexperienced radiologist.\n"
                    "   - Ensure no fixation points are missed during extraction.\n\n"
                    "2. **Compare Fixation Coordinates**:\n"
                    "   - Compare the fixation coordinates of the extracted subgraph A with those in the inexperienced radiologist's subgraph.\n"
                    "   - If any node in subgraph A does not have a matching fixation coordinate in the inexperienced radiologist's subgraph, note this as a missing fixation.\n\n"
                    "3. **Compare Fixation Durations**:\n"
                    "   - For nodes with matching fixation coordinates, compare the fixation durations.\n"
                    "   - If the fixation duration in the inexperienced radiologist's subgraph is shorter, note this as a reduced fixation duration.\n\n"
                    "4. **Determine Undefined Reason**:\n"
                    "   - If neither missing fixation nor reduced fixation duration is detected, note this as an undefined reason.\n\n"
                    f"Here's the extracted subgraph A ({abnormality_A}):\n{subgraph_A}\n"
                    f"and the subgraph {j} from the inexperienced radiologist:\n{subgraph_inexp}.\n\n"
                    "Provide a CORRECT and *VALID* JSON response with the following keys, each set to 0 or 1:\n"
                    "- 'Missing fixation': 1 if any node in the extracted subgraph does not have a matching fixation coordinate in the inexperienced radiologist subgraph, else 0.\n"
                    "- 'Reduced fixation duration': 1 if the fixation duration for a matching node is shorter in the inexperienced radiologist subgraph, else 0.\n"
                    "- 'Undefined reason': 1 if neither of the above conditions are 1, else 0.\n\n"
                    "Do the analysis step by step as described above and provide the CORRECT and *VALID* JSON response. Do not include any code."
                ),
                max_turns=1
            )

            chat_history = response.chat_history
            extracted_json = None

            for message in chat_history:
                if message['role'] == 'user' and message['name'] == agent.name:
                    response_text = message['content']

                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if match:

                        try:
                            extracted_json = json.loads(match.group())
                        except:

                            return 10
                    break

            if extracted_json:
                results_for_A.append(extracted_json)

        print(results_for_A)

        # Combine results for subgraph A using logical OR
        if results_for_A:
            combined_results_A = {
                "Missing fixation": 0,
                "Reduced fixation duration": 0,
                "Undefined reason": 0
            }

            # Extract values for each key from results_for_A
            missing_fixation_values = [
                result["Missing fixation"] for result in results_for_A]
            reduced_fixation_values = [
                result["Reduced fixation duration"] for result in results_for_A]
            undefined_reason_values = [
                result["Undefined reason"] for result in results_for_A]

            combined_results_A = {
                "Missing fixation": int(all(missing_fixation_values)),
                "Reduced fixation duration": int(any(reduced_fixation_values)),
                "Undefined reason": int(any(undefined_reason_values))
            }

            combined_results[abnormality_A] = combined_results_A

    if combined_results:
        discussion_message = (
            "The following combined results have been obtained:\n"
            f"{json.dumps(combined_results, indent=2)}\n\n"
            "Please review the results, perform OR operation for each abnormality, and return the finalized JSON response.\n"
            "Do not include any code and final correct and valid json response should only have three keys and their corrected values (0 or 1):\n"
            "- Missing fixation\n"
            "- Reduced fixation duration\n"
            "- Undefined reason"
            "Do the analysis first and then answer the questions and Do not include any code"

        )

        finalization_response = user_proxy.initiate_chat(
            principal_llm,
            message=discussion_message,
            max_turns=1
        )

        final_decision = None
        for message in finalization_response.chat_history:
            if message['role'] == 'user' and message['name'] == 'PrincipalLLM':
                response_text = message['content']
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:

                    try:
                        final_decision = json.loads(match.group())

                    except:
                        return 10
                break

        if final_decision:
            print("Final Decision after Review:", final_decision)
            return final_decision

    return None


def extract_abnormalities_from_response(response):
    """Parses the response from the Principal LLM to extract missed abnormalities."""

    chat_history = response.chat_history
    missed_abnormalities = []

    for message in chat_history:
        if message['role'] == 'user' and message['name'] == 'PrincipalLLM':
            response_text = message['content'].strip()

            if response_text.upper() == "TERMINATE":
                print("Received 'TERMINATE', stopping extraction.")
                return []

            abnormalities = [
                line.strip().lstrip("-").strip()
                for line in response_text.split("\n")
                if line.strip() and line.strip() != "TERMINATE"
            ]

            if abnormalities:
                missed_abnormalities.extend(abnormalities)
                break

    if missed_abnormalities:
        return missed_abnormalities
    else:
        print("No valid abnormalities extracted.")
        return []


# Function to extract abnormalities from scene graphs


def extract_abnormalities(scene_graph):
    """Extracts the set of abnormalities from a scene graph."""
    # Check if 'subgraphs' exists and handle missing key
    if "subgraphs" not in scene_graph:
        print("Error: 'subgraphs' key missing in scene graph.")
        return set()  # Return an empty set if no subgraphs are found

    return {subgraph["Abnormality"] for subgraph in scene_graph["subgraphs"]}


# Main function to analyze missed findings
def analyze_missed_findings(scene_graphs):
    print("Extracting abnormalities from scene graphs...")

    experienced_abnormalities = extract_abnormalities(
        scene_graphs["experienced"]['scene_graph'])
    inexperienced_abnormalities = extract_abnormalities(
        scene_graphs["inexperienced"]['scene_graph'])

    message = (
        f"Here are two lists of abnormalities:\n"
        f"Experienced abnormalities: {experienced_abnormalities}\n"
        f"Inexperienced abnormalities: {inexperienced_abnormalities}\n"
        "Identify the abnormalities that are present in the experienced list but missing in the inexperienced list. "
        "Please return only the list of abnormalities, in this format:\n"
        "- Abnormality 1\n"
        "- Abnormality 2\n"
        "Do not include any code, descriptive text, or explanations, just the list."
    )

    response = user_proxy.initiate_chat(
        principal_llm, message=message, max_turns=1)

    if not response or not response.chat_history:
        print("Error: Received empty or invalid response from Principal LLM.")
        return {}

    missed_abnormalities = extract_abnormalities_from_response(response)

    if not missed_abnormalities:
        print("No missed abnormalities detected. Terminating further steps.")
        return {}

    print(f"Missed abnormalities detected: {missed_abnormalities}")

    # Step 2: Extract relevant subgraphs based on the missed abnormalities
    subgraphs = [
        subgraph
        for subgraph in scene_graphs["experienced"]['scene_graph']["subgraphs"]
        if subgraph["Abnormality"] in missed_abnormalities
    ]

    if len(subgraphs) == 0:
        print("==========================================")
        print("No matching subgraphs found for the missed abnormalities.")
        print("==========================================")
        return {}

    # Step 3: Recruit agents for comparison
    comparison_results = recruit_agents_for_comparison(
        subgraphs, scene_graphs["inexperienced"])

    if comparison_results == 10:
        return 20

    print(comparison_results)

    print("Analysis complete. Terminating program.")
    return comparison_results


# ---------------------------
# Main processing loop
# ---------------------------
def main():
    resultf = {}

    if os.path.exists(result_output_file):
        with open(result_output_file, 'r') as file:
            resultf = json.load(file)

        print("Loaded prior results of count", len(resultf))

    for K in data.items():
        if K[0] in resultf:
            continue

        start_time = time.time()

        da = data[K[0]]['correct_data']
        result = [
            {
                'FPOGX': da['X_ORIGINAL'][i],
                'FPOGY': da['Y_ORIGINAL'][i],
                'FPOGD': da['FPOGD'][i],
                'Time (in secs)': da['Time (in secs)'][i]
            }
            for i in range(len(da['X_ORIGINAL']))
        ]
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
                'FPOGX': da['X_ORIGINAL'][i],
                'FPOGY': da['Y_ORIGINAL'][i],
                'FPOGD': da['FPOGD'][i],
                'Time (in secs)': da['Time (in secs)'][i]
            }
            for i in range(len(da['X_ORIGINAL']))
        ]
        timestamped_reporti = da['transcript']
        fixation_datai = result

        # Create the scene graph for inexperienced radiologist
        scene_graph_outputi = create_scene_graph_by_sentence(
            timestamped_reporti, fixation_datai)

        scene_graph_exp = scene_graph_output
        scene_graph_inexp = scene_graph_outputi

        scene_graphs = {
            "experienced": scene_graph_exp,
            "inexperienced": scene_graph_inexp
        }

        print(
            "=============================================================================")
        print("Analysis on case id:", K[0])

        report_start_time = time.time()
        missed_findings_report = analyze_missed_findings(scene_graphs)

        if missed_findings_report == 20:
            continue

        if len(missed_findings_report) == 0:
            resultf[K[0]] = {
                "Missing fixation": 0,
                "Reduced fixation duration": 0,
                "Undefined reason": 0,
                "No Missing Subgraph": 1,
            }
        else:
            resultf[K[0]] = missed_findings_report
            resultf[K[0]]["No Missing Subgraph"] = 0

        end_time = time.time()

        resultf[K[0]]['total_time'] = end_time - start_time
        resultf[K[0]]['report_creation_time'] = end_time - report_start_time

        with open(result_output_file, 'w') as f:
            json.dump(resultf, f, indent=2)

        print("=============================================================================\n")

    print(f"MAARTA analysis complete. Results saved to {result_output_file}.")


if __name__ == "__main__":
    main()

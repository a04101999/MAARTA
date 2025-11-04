import numpy as np
import json
import re
import openai
import random
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# ---------------------------
# OpenAI API key configuration
# ---------------------------
openai.api_key = #use key


# ---------------------------
# Data loading
# ---------------------------
with open("original_fixation_transcript_metadata.json", 'r') as file:
    datalab = json.load(file)

with open("original_fixation_transcript_data.json", 'r') as file:
    data = json.load(file)

# Shuffle data randomly
data = {key: data[key] for key in random.sample(data.keys(), len(data))}

# ---------------------------
# Helper functions for scene graph
# ---------------------------
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_closest_times(fixation_data, begin_time, end_time):
    closest_begin_time = max([fix['Time (in secs)'] for fix in fixation_data if fix['Time (in secs)'] <= begin_time], default=None)
    closest_end_time = min([fix['Time (in secs)'] for fix in fixation_data if fix['Time (in secs)'] >= end_time], default=None)
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
            dist = euclidean_distance(previous_node["fixation_point"], current_node["fixation_point"])
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
        closest_begin_time, closest_end_time = find_closest_times(fixation_data, begin_time, end_time)
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

        subgraph = create_subgraph(sentence_text, fixation_nodes, sentence_counter)
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
manager = GroupChatManager(name="RadiologyComparisonManager", groupchat=group_chat)

# ---------------------------
# Helper functions for MAARTA
# ---------------------------
def extract_abnormalities(scene_graph):
    if "subgraphs" not in scene_graph:
        return set()
    return {subgraph["Abnormality"] for subgraph in scene_graph["subgraphs"]}

def extract_abnormalities_from_response(response):
    chat_history = response.chat_history
    missed_abnormalities = []
    for message in chat_history:
        if message['role'] == 'user' and message['name'] == 'PrincipalLLM':
            response_text = message['content'].strip()
            if response_text.upper() == "TERMINATE":
                return []
            abnormalities = [line.strip().lstrip("-").strip() for line in response_text.split("\n") if line.strip() and line.strip() != "TERMINATE"]
            if abnormalities:
                missed_abnormalities.extend(abnormalities)
                break
    return missed_abnormalities

# ---------------------------
# Core MAARTA functions
# ---------------------------
def recruit_agents_for_comparison(subgraphs, scene_graph_inexp):
    if not subgraphs:
        return None
    if "scene_graph" not in scene_graph_inexp or "subgraphs" not in scene_graph_inexp["scene_graph"]:
        return None
    inexp_subgraphs = scene_graph_inexp["scene_graph"]["subgraphs"]
    combined_results = {}

    for i, subgraph_A in enumerate(subgraphs):
        abnormality_A = subgraph_A["Abnormality"]
        comparison_agents = {
            f"ComparisonAgent_{i}_{j}": AssistantAgent(
                name=f"ComparisonAgent_{i}_{j}",
                description=f"Compares subgraph A ({abnormality_A}) with subgraph {j} from the inexperienced radiologist.",
                llm_config={"model": "gpt-4o-mini", "api_key": openai.api_key}
            )
            for j in range(len(inexp_subgraphs))
        }
        manager.groupchat.agents.extend(comparison_agents.values())
        results_for_A = []

        for j, subgraph_inexp in enumerate(inexp_subgraphs):
            agent = comparison_agents.get(f"ComparisonAgent_{i}_{j}")
            if not agent:
                continue
            response = user_proxy.initiate_chat(
                agent,
                message=(
                    f"Let's perform the analysis step by step:\n"
                    f"Analyze subgraph A ({abnormality_A}) vs subgraph {j}.\n"
                    "Provide a CORRECT and VALID JSON response with keys 'Missing fixation', 'Reduced fixation duration', 'Undefined reason'."
                ),
                max_turns=1
            )

            chat_history = response.chat_history
            extracted_json = None
            for message in chat_history:
                if message['role'] == 'user' and message['name'] == agent.name:
                    match = re.search(r'\{.*\}', message['content'], re.DOTALL)
                    if match:
                        try:
                            extracted_json = json.loads(match.group())
                        except:
                            return 10
                    break
            if extracted_json:
                results_for_A.append(extracted_json)

        if results_for_A:
            combined_results_A = {
                "Missing fixation": int(all(result["Missing fixation"] for result in results_for_A)),
                "Reduced fixation duration": int(any(result["Reduced fixation duration"] for result in results_for_A)),
                "Undefined reason": int(any(result["Undefined reason"] for result in results_for_A))
            }
            combined_results[abnormality_A] = combined_results_A

    if combined_results:
        discussion_message = (
            f"The following combined results have been obtained:\n{json.dumps(combined_results, indent=2)}\n"
            "Please review and provide the finalized JSON response."
        )
        finalization_response = user_proxy.initiate_chat(principal_llm, message=discussion_message, max_turns=1)
        final_decision = None
        for message in finalization_response.chat_history:
            if message['role'] == 'user' and message['name'] == 'PrincipalLLM':
                match = re.search(r'\{.*\}', message['content'], re.DOTALL)
                if match:
                    try:
                        final_decision = json.loads(match.group())
                    except:
                        return 10
                break
        if final_decision:
            return final_decision
    return None

def analyze_missed_findings(scene_graphs):
    experienced_abnormalities = extract_abnormalities(scene_graphs["experienced"]['scene_graph'])
    inexperienced_abnormalities = extract_abnormalities(scene_graphs["inexperienced"]['scene_graph'])

    message = (
        f"Experienced abnormalities: {experienced_abnormalities}\n"
        f"Inexperienced abnormalities: {inexperienced_abnormalities}\n"
        "Identify abnormalities present in experienced but missing in inexperienced."
    )
    response = user_proxy.initiate_chat(principal_llm, message=message, max_turns=1)
    if not response or not response.chat_history:
        return {}

    missed_abnormalities = extract_abnormalities_from_response(response)
    if not missed_abnormalities:
        return {}

    subgraphs = [sg for sg in scene_graphs["experienced"]['scene_graph']["subgraphs"] if sg["Abnormality"] in missed_abnormalities]
    if not subgraphs:
        return {}

    comparison_results = recruit_agents_for_comparison(subgraphs, scene_graphs["inexperienced"])
    if comparison_results == 10:
        return 20

    return comparison_results

# ---------------------------
# Main processing loop
# ---------------------------
resultf = {}
for K in data.items():
    da = data[K[0]]['correct_data'] if len(data[K[0]]['incorrect_data']) == 0 else data[K[0]]['incorrect_data']
    result = [{"FPOGX": da['X_ORIGINAL'][i], "FPOGY": da['Y_ORIGINAL'][i], "FPOGD": da['FPOGD'][i], "Time (in secs)": da['Time (in secs)'][i]} for i in range(len(da['X_ORIGINAL']))]

    timestamped_report = da['transcript']
    fixation_data = result

    scene_graph_output = create_scene_graph_by_sentence(timestamped_report, fixation_data)
    scene_graph_exp = scene_graph_output
    scene_graph_inexp = create_scene_graph_by_sentence(data[K[0]]['incorrect_data']['transcript'] if len(data[K[0]]['incorrect_data']) > 0 else data[K[0]]['correct_data']['transcript'], fixation_data)

    scene_graphs = {"experienced": scene_graph_exp, "inexperienced": scene_graph_inexp}

    missed_findings_report = analyze_missed_findings(scene_graphs)
    if missed_findings_report == 20:
        continue

    resultf[K[0]] = missed_findings_report

# Save results to a JSON file
with open("missed_findings_results.json", 'w') as f:
    json.dump(resultf, f, indent=2)

print("MAARTA analysis complete. Results saved to 'missed_findings_results.json'.")

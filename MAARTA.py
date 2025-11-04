
import json
import re
import openai
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# ------------------------------
# OpenAI API Key
# ------------------------------
openai.api_key = #get key

# ------------------------------
# Agents
# ------------------------------
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


# ------------------------------
# Utility Functions
# ------------------------------
def extract_abnormalities(scene_graph):
    """Extracts the set of abnormalities from a scene graph."""
    if "subgraphs" not in scene_graph:
        print("Error: 'subgraphs' key missing in scene graph.")
        return set()
    return {subgraph["Abnormality"] for subgraph in scene_graph["subgraphs"]}


def extract_abnormalities_from_response(response):
    """Parses the response from the Principal LLM to extract missed abnormalities."""
    chat_history = response.chat_history
    missed_abnormalities = []

    for message in chat_history:
        if message['role'] == 'user' and message['name'] == 'PrincipalLLM':
            text = message['content'].strip()
            if text.upper() == "TERMINATE":
                return []

            abnormalities = [
                line.strip().lstrip("-").strip()
                for line in text.split("\n")
                if line.strip() and line.strip() != "TERMINATE"
            ]
            if abnormalities:
                missed_abnormalities.extend(abnormalities)
                break

    return missed_abnormalities


# ------------------------------
# Core MAARTA Functions
# ------------------------------
def recruit_agents_for_comparison(subgraphs, scene_graph_inexp):
    """Creates comparison agents for each missed abnormality and compares with student subgraphs."""
    if not subgraphs:
        return None

    inexp_subgraphs = scene_graph_inexp.get("scene_graph", {}).get("subgraphs")
    if not inexp_subgraphs:
        return None

    combined_results = {}

    for i, subgraph_A in enumerate(subgraphs):
        abnormality_A = subgraph_A["Abnormality"]

        # Create agents for this subgraph
        comparison_agents = {
            f"ComparisonAgent_{i}_{j}": AssistantAgent(
                name=f"ComparisonAgent_{i}_{j}",
                description=f"Compares subgraph A ({abnormality_A}) with subgraph {j} from the student.",
                llm_config={"model": "gpt-4o-mini", "api_key": openai.api_key}
            )
            for j in range(len(inexp_subgraphs))
        }

        manager.groupchat.agents.extend(comparison_agents.values())
        results_for_A = []

        # Perform comparisons
        for j, subgraph_inexp in enumerate(inexp_subgraphs):
            agent = comparison_agents.get(f"ComparisonAgent_{i}_{j}")
            if not agent:
                continue

            message = (
                f"Let's perform the analysis step by step:\n\n"
                "1. Extract fixation points from subgraph A ({abnormality_A}) "
                "and subgraph {j} from the student.\n"
                "2. Compare fixation coordinates.\n"
                "3. Compare fixation durations.\n"
                "4. Determine undefined reason.\n\n"
                f"Subgraph A: {subgraph_A}\nSubgraph {j}: {subgraph_inexp}\n"
                "Provide CORRECT JSON response with keys: "
                "'Missing fixation', 'Reduced fixation duration', 'Undefined reason' (0 or 1)."
            )

            response = user_proxy.initiate_chat(agent, message=message, max_turns=1)
            extracted_json = None

            for msg in response.chat_history:
                if msg['role'] == 'user' and msg['name'] == agent.name:
                    match = re.search(r'\{.*\}', msg['content'], re.DOTALL)
                    if match:
                        try:
                            extracted_json = json.loads(match.group())
                        except:
                            return 10
                    break

            if extracted_json:
                results_for_A.append(extracted_json)

        # Combine results for subgraph A
        if results_for_A:
            combined_results[abnormality_A] = {
                "Missing fixation": int(all(r["Missing fixation"] for r in results_for_A)),
                "Reduced fixation duration": int(any(r["Reduced fixation duration"] for r in results_for_A)),
                "Undefined reason": int(any(r["Undefined reason"] for r in results_for_A)),
            }

    # Principal LLM final review
    if combined_results:
        discussion_message = (
            f"The following combined results have been obtained:\n{json.dumps(combined_results, indent=2)}\n"
            "Please finalize the JSON response with keys: "
            "'Missing fixation', 'Reduced fixation duration', 'Undefined reason'."
        )

        final_response = user_proxy.initiate_chat(principal_llm, message=discussion_message, max_turns=1)
        final_decision = None

        for msg in final_response.chat_history:
            if msg['role'] == 'user' and msg['name'] == 'PrincipalLLM':
                match = re.search(r'\{.*\}', msg['content'], re.DOTALL)
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
    """High-level function to detect missed findings and compare them using MAARTA."""
    experienced_abnormalities = extract_abnormalities(scene_graphs["experienced"]['scene_graph'])
    inexperienced_abnormalities = extract_abnormalities(scene_graphs["inexperienced"]['scene_graph'])

    message = (
        f"Experienced abnormalities: {experienced_abnormalities}\n"
        f"Inexperienced abnormalities: {inexperienced_abnormalities}\n"
        "Identify abnormalities present in experienced but missing in inexperienced. "
        "Return only a list of abnormalities."
    )

    response = user_proxy.initiate_chat(principal_llm, message=message, max_turns=1)
    missed_abnormalities = extract_abnormalities_from_response(response)
    if not missed_abnormalities:
        return {}

    subgraphs = [
        sg for sg in scene_graphs["experienced"]['scene_graph']["subgraphs"]
        if sg["Abnormality"] in missed_abnormalities
    ]

    if not subgraphs:
        return {}

    comparison_results = recruit_agents_for_comparison(subgraphs, scene_graphs["inexperienced"])
    if comparison_results == 10:
        return 20

    return comparison_results

import numpy as np


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""

    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_closest_times(fixation_data, begin_time, end_time):
    """Find the closest fixation times to the begin and end times of the phrase."""

    closest_begin_time = max([fix['Time (in secs)']
                             for fix in fixation_data if fix['Time (in secs)'] <= begin_time], default=None)
    closest_end_time = min([fix['Time (in secs)']
                           for fix in fixation_data if fix['Time (in secs)'] >= end_time], default=None)
    return closest_begin_time, closest_end_time


def create_subgraph(sentence_text, fixation_nodes, sentence_counter):
    """Creates a subgraph for a given set of fixation nodes, including revisit edges."""

    subgraph = {
        "Abnormality": sentence_text,
        "nodes": fixation_nodes,
        "edges": []
    }
    # Dictionary to track nodes by their positions for revisits
    position_tracker = {}

    # Create edges based on Euclidean distance and check for revisits
    for i, current_node in enumerate(fixation_nodes):
        # Track revisits: Check if position was already seen
        position_key = tuple(current_node["fixation_point"])
        if position_key in position_tracker:
            original_node = position_tracker[position_key]
            # Add a revisit edge from the original node to the current revisited node
            subgraph["edges"].append({
                "from": original_node["id"],
                "to": current_node["id"],
                "type": "revisit"
            })
            # print(f"Revisit detected: Edge added from {original_node['id']} to {current_node['id']}.")
        else:
            # If it's a new position, add it to the tracker
            position_tracker[position_key] = current_node

        # Create sequential edges based on Euclidean distance for immediate neighbors
        if i > 0:
            previous_node = fixation_nodes[i - 1]
            dist = euclidean_distance(
                previous_node["fixation_point"], current_node["fixation_point"])
            subgraph["edges"].append({
                "from": previous_node["id"],
                "to": current_node["id"],
                "distance": round(dist, 3)
            })
    # Debugging output
    # if fixation_nodes:
      #  print(f"Subgraph for '{sentence_text}' has {len(fixation_nodes)} nodes and {len(subgraph['edges'])} edges.")
    # else:
       # print(f"No nodes added to subgraph for '{sentence_text}'.")

    return subgraph


def create_scene_graph_by_sentence(timestamped_report, fixation_data):
    """Construct a scene graph with each sentence as a phrase and idle fixations as a separate subgraph."""
    scene_graph = {
        "scene_graph": {
            "subgraphs": [],
            "inter_phrase_edges": []
        }
    }

    sentence_counter = 1  # Unique ID base for each sentence
    phrase_time_ranges = []  # List to store phrase time ranges for idle fixation exclusion
    idle_fixations = []  # List to collect fixations with no phrase association

    # Process each sentence in the report
    for sentence_info in timestamped_report:
        sentence_text = sentence_info['sentence']
        begin_time, end_time = sentence_info['begin_time'], sentence_info['end_time']
        # Record time range for this phrase
        phrase_time_ranges.append((begin_time, end_time))
        # Get the closest begin and end fixation times
        closest_begin_time, closest_end_time = find_closest_times(
            fixation_data, begin_time, end_time)
        if closest_begin_time is None or closest_end_time is None:
            # print(f"No fixations found within bounds for '{sentence_text}'")
            continue

        # Collect fixation points within this phrase's time range
        fixation_nodes = []
        for fixation in fixation_data:
            fixation_time = fixation['Time (in secs)']
            if closest_begin_time <= fixation_time <= closest_end_time:
                fpx, fpy, fpd = fixation['X_ORIGINAL'], fixation['Y_ORIGINAL'], fixation['FPOGD']
                # Unique node ID
                node_id = f"fixation_{sentence_counter * 10 + len(fixation_nodes)}"
                fixation_nodes.append({
                    "id": node_id,
                    "fixation_point": [fpx, fpy],
                    "fixation_duration": fpd
                })

        # Create and add subgraph for the phrase, including revisits
        subgraph = create_subgraph(
            sentence_text, fixation_nodes, sentence_counter)
        scene_graph["scene_graph"]["subgraphs"].append(subgraph)

        # Connect to previous sentence if applicable
        if len(scene_graph["scene_graph"]["subgraphs"]) > 1:
            previous_phrase = scene_graph["scene_graph"]["subgraphs"][-2]["Abnormality"]
            scene_graph["scene_graph"]["inter_phrase_edges"].append({
                "from": previous_phrase,
                "to": sentence_text,
                "relationship": "sequence"
            })

        sentence_counter += 1  # Increment for unique node IDs

    # Collect idle fixations (not within any phrase's time bounds)
    for fixation in fixation_data:
        fixation_time = fixation['Time (in secs)']
        if not any(begin <= fixation_time <= end for begin, end in phrase_time_ranges):
            node_id = f"idle_fixation_{len(idle_fixations)}"
            idle_fixations.append({
                "id": node_id,
                "fixation_point": [fixation['X_ORIGINAL'], fixation['Y_ORIGINAL']],
                "fixation_duration": fixation['FPOGD']
            })

    # Create and add idle subgraph
    idle_subgraph = create_subgraph("Idle Subgraph", idle_fixations, 0)
    scene_graph["scene_graph"]["subgraphs"].append(idle_subgraph)

    return scene_graph


def compare_scene_graphs_with_llm(scene_graph_exp, scene_graph_inexp):
    # Initialize result dictionary
    result = {
        "No Missing Subgraph": 0,
        "Missing fixation": 0,
        "reduced fixation duration": 0,
        "no reason": 0
    }

    # Initialize lists to save missed details
    missed_subgraphs = []
    missed_fixation_points = []
    nodes_with_reduced_fixation_duration = []
    subgraphs_missed_no_reason = []

    # Extract subgraphs
    subgraphs_exp = scene_graph_exp.get("scene_graph", {}).get("subgraphs", [])
    subgraphs_inexp = scene_graph_inexp.get(
        "scene_graph", {}).get("subgraphs", [])

    # Track missing subgraphs
    missing_subgraphs = []

    # Compare each subgraph in Scene Graph 1 (Experienced) with Scene Graph 2 (Inexperienced)
    for subgraph_exp in subgraphs_exp:
        # Check if subgraph is in Scene Graph 2 by comparing abnormalities
        abnormality_exp = subgraph_exp["Abnormality"]
        matched_subgraph_inexp = None

        for subgraph_inexp in subgraphs_inexp:
            if subgraph_inexp["Abnormality"] == abnormality_exp:
                matched_subgraph_inexp = subgraph_inexp
                break

        if matched_subgraph_inexp is None and abnormality_exp != "Idle Subgraph":
            # If abnormality subgraph is missing in Scene Graph 2, add it to missing_subgraphs
            missing_subgraphs.append(subgraph_exp)
            # Save missed subgraph details
            missed_subgraphs.append(subgraph_exp)

    # Set "No Missing Subgraph" if no missing subgraphs are found
    if not missing_subgraphs:
        result["No Missing Subgraph"] = 1
    else:
        # For each missing subgraph, check for missing fixation points
        for missing_subgraph in missing_subgraphs:
            found_fixation_points = False
            all_nodes_present = True

            for node_exp in missing_subgraph["nodes"]:
                matched_node = False
                for subgraph_inexp in subgraphs_inexp:
                    for node_inexp in subgraph_inexp["nodes"]:
                        if node_exp["fixation_point"] == node_inexp["fixation_point"]:
                            found_fixation_points = True
                            matched_node = True
                            # Check fixation duration for reduced duration
                            if node_inexp["fixation_duration"] < node_exp["fixation_duration"]:
                                result["reduced fixation duration"] = 1
                                nodes_with_reduced_fixation_duration.append(
                                    node_exp)  # Save node with reduced duration
                # If any node from missing subgraph is not found in Scene Graph 2
                if not matched_node:
                    all_nodes_present = False
                    # Save missed fixation point details
                    missed_fixation_points.append(node_exp)
            if found_fixation_points:
                result["Missing fixation"] = 1
            if all_nodes_present and result["reduced fixation duration"] == 0:
                result["no reason"] = 1
                # Save subgraph missed with no reason
                subgraphs_missed_no_reason.append(missing_subgraph)

    # Special handling for "Idle Subgraph"
    idle_subgraph_exp = next(
        (sg for sg in subgraphs_exp if sg["Abnormality"] == "Idle Subgraph"), None)
    idle_subgraph_inexp = next(
        (sg for sg in subgraphs_inexp if sg["Abnormality"] == "Idle Subgraph"), None)

    if idle_subgraph_exp and idle_subgraph_inexp:
        # Compare fixation points and durations specifically for Idle Subgraph
        for node_exp in idle_subgraph_exp["nodes"]:
            matched_node = False
            for node_inexp in idle_subgraph_inexp["nodes"]:
                if node_exp["fixation_point"] == node_inexp["fixation_point"]:
                    matched_node = True
                    # Check fixation duration
                    if node_inexp["fixation_duration"] < node_exp["fixation_duration"]:
                        result["reduced fixation duration"] = 1
                        nodes_with_reduced_fixation_duration.append(
                            node_exp)  # Save node with reduced duration
            if not matched_node:
                result["Missing fixation"] = 1
                # Save missed fixation point details
                missed_fixation_points.append(node_exp)
    elif idle_subgraph_exp and not idle_subgraph_inexp:
        # If Idle Subgraph is missing entirely in Scene Graph 2
        result["Missing fixation"] = 1
        # Save missing Idle Subgraph details
        missed_subgraphs.append(idle_subgraph_exp)

    # Save all collected lists to result
    result["missed_subgraphs"] = missed_subgraphs
    result["missed_fixation_points"] = missed_fixation_points
    result["nodes_with_reduced_fixation_duration"] = nodes_with_reduced_fixation_duration
    # Step 4: Create the prompt for LLM with context and logical corrections
    result["subgraphs_missed_no_reason"] = subgraphs_missed_no_reason
    prompt = f"""
        The following are two scene graphs representing the analyses of the same medical image by two radiologists. "Scene Graph 1" corresponds to the experienced radiologist, and "Scene Graph 2" corresponds to the inexperienced radiologist.
        
        Scene Graph 1 (Experienced Radiologist):
        {scene_graph_exp}
        
        Scene Graph 2 (Inexperienced Radiologist):
        {scene_graph_inexp}
        
        You have been provided with a list of observed differences between these two scene graphs of an experienced radiologist and that of an inexperienced radiologist. Based on these differences, please analyze and reason through the findings to generate a JSON response that reflects the nature of the discrepancies.
        
        The differences are provided in the following format:
        - List of missing subgraphs: {missed_subgraphs}
        - List of missing fixation points: {missed_fixation_points}
        - List of nodes with reduced fixation duration: {nodes_with_reduced_fixation_duration}
        
        Based on these observations, your task is to return a JSON object with the following keys, assigning each a value of 0 or 1:
        
        - "No Missing Subgraph": Set this to 1 if there are no missing subgraphs in the list. Otherwise, set it to 0.
        - "Missing Subgraph due to Missing fixation": Set to 1 if there are any missing fixation points, otherwise set it to 0.
        - "Missing Subgraph due to reduced fixation duration": Set to 1 if any fixation duration is shorter than expected, otherwise set it to 0.
        - "Missing Subgraph due to undefined reason": Set to 1 only if there are missing subgraphs with no clear reason (i.e., no missing fixation points or reduced durations). Set it to 0 if there are clear reasons (such as missing fixation points or reduced durations).
        
        Please provide your reasoning for each point before returning the JSON response.
        """

    return prompt

from collections import deque
from datetime import datetime
from typing import List, Dict, Any
import json

import numpy as np

from lpm_kernel.L1.bio import Cluster
import logging


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_cur_time() -> str:
    """
    Returns the current time formatted as a string.
    
    Returns:
        str: Current time formatted according to TIME_FORMAT.
    """
    cur_time = datetime.now().strftime(TIME_FORMAT)
    return cur_time


def find_connected_components(
    cluster_list: List[Cluster], cluster_merge_distance: float
) -> List[List[Cluster]]:
    """
    Finds connected components in a list of clusters based on a distance threshold.
    
    Args:
        cluster_list: List of Cluster objects to analyze.
        cluster_merge_distance: Maximum distance for clusters to be considered connected.
        
    Returns:
        List[List[Cluster]]: List of connected components, where each component is a list of clusters.
    """
    adjacency_matrix = np.array(
        [
            [
                np.linalg.norm(cluster1.cluster_center - cluster2.cluster_center)
                for cluster2 in cluster_list
            ]
            for cluster1 in cluster_list
        ]
    )

    cluster_n = len(cluster_list)
    visited = [False] * cluster_n
    components = []

    def bfs(start: int):
        queue = deque([start])
        component = []
        visited[start] = True

        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in range(cluster_n):
                if (
                    not visited[neighbor]
                    and adjacency_matrix[node, neighbor] < cluster_merge_distance
                ):
                    visited[neighbor] = True
                    queue.append(neighbor)
        return component

    for i in range(cluster_n):
        if not visited[i]:
            components.append(bfs(i))

    return [[cluster_list[i] for i in component] for component in components]


def is_valid_note(note: Dict[str, Any]) -> bool:
    """
    Checks if a note contains valid creation time information.
    
    Args:
        note: Dictionary containing note data.
        
    Returns:
        bool: True if the note has a valid creation time, False otherwise.
    """
    if "createTime" in note and note["createTime"]:
        return True
    return False


def is_valid_todo(todo: Dict[str, Any]) -> bool:
    """
    Checks if a todo item contains valid creation time information.
    
    Args:
        todo: Dictionary containing todo data.
        
    Returns:
        bool: True if the todo has a valid creation time, False otherwise.
    """
    if "createTime" in todo and todo["createTime"]:
        return True
    return False


def is_valid_chat(chat: Dict[str, Any]) -> bool:
    """
    Checks if a chat contains valid creation time and summary information.
    
    Args:
        chat: Dictionary containing chat data.
        
    Returns:
        bool: True if the chat has valid creation time and summary, False otherwise.
    """
    if (
        "createTime" in chat
        and chat["createTime"]
        and "summary" in chat
        and chat["summary"]
    ):
        return True
    return False


def save_true_topics(true_topics_res: Dict[str, Dict], topics_path: str) -> None:
    """
    Save topics clustering results to a JSON file, excluding embedding data.

    Args:
        true_topics_res: Dictionary containing topic clustering results.
        topics_path: Path to save the JSON file.
    """
    # Create a copy to avoid modifying original
    topics_to_save = {}

    for cluster_id, cluster_data in true_topics_res.items():
        # Create new cluster dict without embeddings
        topics_to_save[cluster_id] = {
            "indices": cluster_data["indices"],
            "docIds": cluster_data["docIds"],
            "contents": cluster_data["contents"],
            "chunkIds": cluster_data["chunkIds"],
            "tags": cluster_data["tags"],
            "topic": cluster_data["topic"],
            "topicId": cluster_data["topicId"],
            "recTimes": cluster_data["recTimes"],
        }

    # Save to JSON file
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics_to_save, f, ensure_ascii=False, indent=4)

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_max_doc_id_length(entity_map):
    """Gets the maximum document ID length and corresponding entity.
    
    Args:
        entity_map: List of entity dictionaries containing doc_id lists
        
    Returns:
        tuple: (max_length, max_length_entity) where max_length is the maximum
               number of document IDs and max_length_entity is the entity with
               the most document IDs
    """
    max_length = 0
    max_length_entity = None
    
    for entity in entity_map:
        doc_ids = entity.get("doc_id", [])
        if len(doc_ids) > max_length:
            max_length = len(doc_ids)
            max_length_entity = entity
    
    return max_length, max_length_entity


def save_to_json(data, filename):
    """Saves the parsed JSON data to a file.
    
    Args:
        data: The data to be saved as JSON
        filename: Path to the output file
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving to file: {e}")


def map_doc_id_length_to_needs_count(doc_id_length, max_doc_id_length, min_needs, max_needs):
    """Maps the doc_id length to a number between min_needs and max_needs using linear interpolation.
    
    Args:
        doc_id_length: Length of document ID list to map
        max_doc_id_length: Maximum document ID length in the dataset
        min_needs: Minimum needs count
        max_needs: Maximum needs count
        
    Returns:
        int: Mapped needs count value
    """
    # Ensure doc_id_length is at least 1
    doc_id_length = max(1, doc_id_length)
    
    # Handle the case where max_doc_id_length is 1 to prevent division by zero
    if max_doc_id_length <= 1:
        return min_needs  # Return min_needs when there's only one document or invalid input
    
    # Linear mapping formula: y = (x - x1)(y2 - y1)/(x2 - x1) + y1
    mapped_value = (doc_id_length - 1) * (max_needs - min_needs) / (max_doc_id_length - 1) + min_needs
    
    # Round to nearest integer and ensure it's within bounds
    return min(max(round(mapped_value), min_needs), max_needs)


def map_doc_id_lengths_parallel(doc_id_lengths, max_doc_id_length, min_needs, max_needs, num_threads=2):
    """Maps multiple doc_id_length values to needs counts in parallel using multithreading.
    
    Args:
        doc_id_lengths: List of doc_id_length values to process
        max_doc_id_length: Maximum document ID length
        min_needs: Minimum needs count
        max_needs: Maximum needs count
        num_threads: Number of threads to use (default: 2)
        
    Returns:
        list: List of mapped needs counts corresponding to the input doc_id_lengths
    """
    # Define a wrapper function for the ThreadPoolExecutor
    def process_single_doc_id(args):
        idx, doc_id_length = args
        result = map_doc_id_length_to_needs_count(doc_id_length, max_doc_id_length, min_needs, max_needs)
        return idx, result
    
    # Create a list of (index, doc_id_length) tuples for processing
    indexed_inputs = list(enumerate(doc_id_lengths))
    
    # Use ThreadPoolExecutor to process in parallel
    results = [None] * len(doc_id_lengths)
    with ThreadPoolExecutor(max_workers=min(num_threads, len(doc_id_lengths))) as executor:
        futures = [executor.submit(process_single_doc_id, item) for item in indexed_inputs]
        
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error(f"Error processing doc_id_length: {e}")
    
    return results


def multi_process_request(all_messages, max_workers, func, structure=None):
    """Processes multiple requests in parallel using ThreadPoolExecutor.
    
    Args:
        all_messages: List of messages to process
        max_workers: Maximum number of worker threads
        func: Function to apply to each message
        structure: Optional structure parameter to pass to the function
        
    Returns:
        list: Results from processing each message
    """
    with ThreadPoolExecutor(max_workers=min(max_workers, len(all_messages))) as executor:
        futures = [(i, executor.submit(func, messages, structure)) if structure is not None else (i, executor.submit(func, messages)) for i, messages in enumerate(all_messages)]
        results = [None] * len(all_messages)  

        for i, future in tqdm(futures):
            try:
                result = future.result()
                results[i] = result  
            except Exception as e:
                results[i] = f"Raise ERROR: {e} WHEN GENERATE RESPONSE"
    return results
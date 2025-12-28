from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import ast
import concurrent.futures
import json
import jsonlines
import logging
import os
import random
import traceback

from openai import OpenAI
from tqdm import tqdm

from lpm_kernel.L1.bio import Note
from lpm_kernel.L2.data_pipeline.data_prep.context_data.context_config import enc, needs_dict, min_needs_count, max_needs_count
from lpm_kernel.L2.data_pipeline.data_prep.context_data.prompt import (
    needs_prompt_v1, context_enhance_prompt_zh, context_enhance_prompt_en,
    find_related_note_todos__SYS_ZH, find_related_note_todos__SYS_EN,
    expert_response_prompt, coarse_grained_prompt_a, coarse_grained_prompt_b,
    fine_grained_prompt_a, fine_grained_prompt_b, fine_grained_prompt_c
)
from lpm_kernel.L2.data_pipeline.data_prep.context_data.utils import (
    get_max_doc_id_length, save_to_json, map_doc_id_length_to_needs_count, multi_process_request
)
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config


def parse_model_response(response_content: str) -> List[str]:
    """
    Parse the response content from a model and extract topics.
    
    Args:
        response_content: The raw response content from a model

    Returns:
        A list of topic strings extracted from the response
    """
    try:
        # Attempt to parse the response content as JSON
        response_dict = json.loads(response_content)
        # Extract the topics list from the parsed JSON
        topics = response_dict.get("topics", [])
        return topics
    except json.JSONDecodeError:
        # If parsing fails, return an empty list
        logging.error("Error parsing JSON. Response content is not valid JSON.")
        return []


class ContextGenerator:
    """
    Generates context data by processing notes and creating enhanced content.
    """

    def __init__(self, preferred_language: str = "English", user_name: str = "", user_bio: str = ""):
        """
        Initialize the ContextGenerator with user preferences and configuration.
        
        Args:
            preferred_language: The preferred language for generated content
            user_name: The name of the user
            user_bio: The biography information of the user
        """
        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        if user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.model_name = user_llm_config.chat_model_name
    
            self.client = OpenAI(
                api_key=user_llm_config.api_key,
                base_url=user_llm_config.endpoint,
            )
        self.preferred_language = preferred_language
        self.critic_checkpoint_path = "./critic_task_checkpoint.json"
        
        self.multi_time = 1
        self.user_name = user_name
        self.user_bio = user_bio


    def get_notes_content(self, entity_json: Dict, 
                      note_list: List[Note],
                      max_tokens: int = 20000) -> str:
        """
        Get content from notes referenced in the entity JSON.
        
        Args:
            entity_json: A dictionary containing document IDs
            note_list: A list of Note objects
            max_tokens: Maximum number of tokens to include in the output
            
        Returns:
            A string containing the formatted note content
        """
        note_json = {}
        for note in note_list:
            note_json[note.id] = note.to_json()

        # get notes content
        total_tokens = 0
        notes_content = []
        already_content = []
        for doc_id in entity_json["doc_id"]:
            new_content = self.format_note(note_json[doc_id])
            if new_content in already_content:
                continue
            already_content.append(new_content)
            content_tokens = len(enc.encode(new_content))
            
            # The condition here may actually cause a slight overflow, but it's acceptable
            if total_tokens + content_tokens > max_tokens:
                break
            total_tokens += content_tokens
            notes_content.append(new_content)
        return "\n".join(notes_content)


    def format_note(self, note_json: Dict) -> str:
        """
        Format a note as a string based on the preferred language.
        
        Args:
            note_json: A dictionary containing note data
            
        Returns:
            A formatted string representing the note
        """
        if self.preferred_language == "English":
            return f"Note:\nTitle: {note_json['title'] or ''}\nContent: {note_json['content'] or ''}\nKey Points: {note_json['insight'] or ''}\n"
        else:
            return f"Note:\nTitle: {note_json['title'] or ''}\nContent: {note_json['content'] or ''}\nKey Points: {note_json['insight'] or ''}\n"


    def _generate_needs(self, needs_prompt_content: str, entity_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate need responses using the model.
        
        Args:
            needs_prompt_content: The prompt to send to the model
            entity_name: The name of the entity
            
        Returns:
            A tuple containing the model response and entity name, or (None, None) if there's an error
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": needs_prompt_content}],
                temperature=1,
                max_tokens=3000,
            )
            logging.info(f"Response content: {response.choices[0].message.content}")
            return response.choices[0].message.content, entity_name
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode API response as JSON: {e}")
            return None, None
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error generating needs response: {e}")
            return None, None


    def generate_context_needs(self, note_list: List[Note], entity_map_path: str, data_output_base_dir: str, needs_file_name: str) -> None:
        """
        Generate context needs based on note list and entity map.
        
        Args:
            note_list: A list of Note objects
            entity_map_path: Path to the entity map file
            data_output_base_dir: Directory for output data
            needs_file_name: Filename for the output needs file
        """
        with open(entity_map_path, 'r') as f:
            entity_map = json.load(f)
        
        max_length, max_entity = get_max_doc_id_length(entity_map)
        logging.info(f"Maximum doc_id length: {max_length}")

        if max_entity:
            logging.info(f"Entity with max doc_ids: {max_entity['entity_name']}")
        
        selected_needs = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for entity in tqdm(entity_map, desc="Processing entities"):
                doc_id_length = len(entity.get("doc_id", []))
                needs_count = map_doc_id_length_to_needs_count(
                    doc_id_length, 
                    max_length,
                    min_needs_count * 1,
                    max_needs_count * 1
                )
                logging.info(f"Entity: {entity['entity_name']}, Doc ID Length: {doc_id_length}, Needs Count: {needs_count}")

                # get notes content
                notes_content = self.get_notes_content(entity, note_list)

                # randomly select needs_count needs from needs_dict with replacement
                for _ in range(needs_count):
                    primary_need = random.choice(list(needs_dict.keys()))
                    secondary_need = random.choice(needs_dict[primary_need])
                    needs_prompt_content = needs_prompt_v1.format(
                        needs=f"{list(secondary_need.keys())[0]}: {list(secondary_need.values())[0]}", 
                        note_content=notes_content, 
                        preferred_language=self.preferred_language
                    )
                    futures.append(executor.submit(self._generate_needs, needs_prompt_content, entity['entity_name']))

            for future in as_completed(futures):
                needs_response, entity_name = future.result()
                if needs_response:
                    selected_needs.append({
                        "needs_response": needs_response,
                        "entity_name": entity_name,
                        "notes_content": notes_content
                    })
                    logging.info(f"length of selected_needs: {len(selected_needs)}")
                else:
                    logging.info(f"Error generating needs response for {primary_need}: {secondary_need}")
        
        save_to_json(selected_needs, data_output_base_dir + "/" + needs_file_name)


    def _extract_initial_needs(self, needs_data: List[Dict]) -> List[str]:
        """
        Extract initial needs expressions from needs data.
        
        Args:
            needs_data: List of dictionaries containing needs data
            
        Returns:
            A list of needs expressions
        """
        # Initialize an empty list to store all needs expressions
        needs_expressions = []
        
        # Iterate through each needs entry
        for need in needs_data:
            # Extract the needs_response field
            needs_response = need.get("needs_response", "{}")
            
            # Convert needs_response from string to dictionary
            try:
                needs_response_dict = json.loads(needs_response)
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON for need: {need}")
                continue  # Skip this entry if parsing fails
            
            # Extract the content of "Needs Expression in User's Tone"
            user_tone_expressions = needs_response_dict.get("Needs Expression in User's Tone", [])
            
            # Add the extracted content to the list
            needs_expressions.extend(user_tone_expressions)
        
        return needs_expressions


    def _process_request(self, messages: List[Dict], format_class: Any = None) -> Optional[str]:
        """
        Process a request to the model with the given messages.
        
        Args:
            messages: List of message dictionaries to send to the model
            format_class: Optional format class for response
            
        Returns:
            The model's response content or None if there's an error
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            return None


    def preprocess4contextEnhance(self, needsAndContext: List[Dict]) -> List[List[Dict]]:
        """
        Preprocess data for context enhancement.
        
        Args:
            needsAndContext: List of dictionaries containing needs and context data
            
        Returns:
            A list of message lists prepared for the model
        """
        processed_data = []
        for item in needsAndContext:
            initial_need = item["initial_need"]
            related_note_todos = item["related_notes"]
            
            # Preprocess related_note_todos
            notes_and_todos = []
            for note_todo in related_note_todos:
                note_str = f"Note Title: {note_todo.get('title', '')}\nNote Content: {note_todo.get('content', '')}\nNote Insight: {note_todo.get('insight', '')}"
                notes_and_todos.append(note_str)
            
            # Combine initial need and related notes/todos
            combined_str = f"Initial Need: {initial_need}\nRelated Notes/Todos:\n" + "\n".join(notes_and_todos)
            
            # Create the message list
            messages = [
                {"role": "system", "content": context_enhance_prompt_zh if self.preferred_language == "Chinese" else context_enhance_prompt_en},
                {"role": "user", "content": combined_str}
            ]
            
            processed_data.append(messages)
        
        return processed_data


    def _send_request(self, messages: List[Dict]) -> str:
        """
        Send a request to the model with the given messages.
        
        Args:
            messages: List of message dictionaries to send to the model
            
        Returns:
            The model's response content or an error message
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Raise ERROR: {e} WHEN GENERATE RESPONSE"


    def _context_enhance(self, needsAndContext: List[Dict], max_workers: int = 10) -> List[str]:
        """
        Enhance context for given needs and context data.
        
        Args:
            needsAndContext: List of dictionaries containing needs and context data
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            A list of enhanced context strings
        """
        processed_data = self.preprocess4contextEnhance(needsAndContext)
        results = multi_process_request(processed_data, max_workers, self._send_request)
        return results


    def _find_related_notes_and_todos(self, initial_needs: List[str], all_notes: List[Dict], 
                                  all_note_str: str, output_file: str) -> List[Dict]:
        """
        Find notes and todos related to each need and save results to a file.
        
        Args:
            initial_needs: List of initial needs
            all_notes: List of all notes
            all_note_str: String representation of all notes
            output_file: Path to the output file
            
        Returns:
            A list of dictionaries containing needs and related notes/todos
        """
        # Find related notes and todos for each need
        all_cot_messages = []
        for query in initial_needs:
            # Select template based on preferred language
            template = find_related_note_todos__SYS_ZH if self.preferred_language == "Chinese" else find_related_note_todos__SYS_EN
            all_cot_messages.append([{
                "role": "user",
                "content": template.format(all_note_str=all_note_str+'\n\n', user_query=query)
            }])

        # Multi-process the COT task
        trying_limit = len(all_cot_messages)
        
        cot_results = multi_process_request(all_cot_messages[:trying_limit], 16, self._process_request)
        all_notes_todos = all_notes
        needsAndRelatedNotesTodos_res = []
        
        for cot_result, need in zip(cot_results, initial_needs):
            try:
                # Try to parse cot_result
                note_todos_ids = ast.literal_eval(cot_result.replace("note_todos_ids: ", ""))
                related_note_todos = [note_todo for note_todo in all_notes_todos if note_todo['id'] in note_todos_ids]
                
                # Simplify the related notes and todos and calculate token count
                simplified_notes_todos, total_tokens = self._simplify_related_notes_todos(related_note_todos)
                
                # Only include the need if the total token count is greater than 50
                if total_tokens > 50:
                    needsAndRelatedNotesTodos_res.append({
                        "initial_need": need,
                        "related_notes": simplified_notes_todos,
                    })
            except (SyntaxError, ValueError) as e:
                logging.error(f"Error parsing cot_result: {cot_result}. Error: {e}")
        
        # Save the results to a file
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(needsAndRelatedNotesTodos_res, file, indent=4, ensure_ascii=False)
        
        return needsAndRelatedNotesTodos_res


    def _clean_and_prepare_data(self, note_file_path: str) -> Tuple[List[Dict], str]:
        """
        Clean and prepare note data from a file.
        
        Args:
            note_file_path: Path to the note data file
            
        Returns:
            A tuple containing the list of cleaned notes and a string representation of all notes
        """
        # Clean and prepare note data
        with open(note_file_path, 'r', encoding='utf-8') as f:
            note_data = json.load(f)
        
        # Remove the "origin_input" field from each note
        for note in note_data:
            note.pop("origin_input", None)
        
        # Save the cleaned note data to a new file
        cleaned_note_file_path = note_file_path.replace('.json', '_cleaned.json')
        with open(cleaned_note_file_path, 'w', encoding='utf-8') as f:
            json.dump(note_data, f, ensure_ascii=False, indent=4)
        
        # Load the cleaned note data
        with open(cleaned_note_file_path, 'r', encoding='utf-8') as file:
            all_notes = json.load(file)
        
        # Prepare string representations of notes
        all_note_str = "\n\n".join([
            f"Note id: {note['id']}, Note title: {note['title']}, Note content: {note['content']}, Note AI Insight: {note.get('insight', '')}"
            for note in all_notes
        ])
        
        return all_notes, all_note_str


    def _clearn_and_prepare_data_from_memory(self, note_list: List[Note]) -> Tuple[List[Dict], str]:
        """
        Clean and prepare note data from a list of Note objects.
        
        Args:
            note_list: A list of Note objects
            
        Returns:
            A tuple containing the list of cleaned notes and a string representation of all notes
        """
        # Clean and prepare note data from a list of Note objects
        note_data = [note.to_json() for note in note_list]
        
        # Remove the "origin_input" field from each note
        for note in note_data:
            note.pop("origin_input", None)
        
        # Prepare string representations of notes
        all_note_str = "\n\n".join([
            f"Note id: {note['id']}, Note title: {note['title']}, Note content: {note['content']}, Note AI Insight: {note.get('insight', '')}"
            for note in note_data
        ])
        
        return note_data, all_note_str


    def _simplify_related_notes_todos(self, related_note_todos: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Simplify related notes and todos by retaining only specific fields and calculate total token count.
        
        Args:
            related_note_todos: List of related notes and todos
            
        Returns:
            A tuple containing the simplified notes/todos and the total token count
        """
        simplified_notes_todos = []
        total_tokens = 0

        for item in related_note_todos:
            simplified_item = {
                'title': item.get('title', ''),
                'content': item.get('content', ''),
                'insight': item.get('insight', '')
            }
            # Concatenate title, content, and insight into a single string
            combined_text = f"{simplified_item['title']} {simplified_item['content']} {simplified_item['insight']}"
            # Calculate token count for the combined string
            token_count = len(enc.encode(combined_text))
            
            simplified_notes_todos.append(simplified_item)
            total_tokens += token_count

        return simplified_notes_todos, total_tokens


    def generate_context_enhance_data(self, data_output_base_dir: str, needs_file_name: str, 
                                   context_enhanced_res_file_name: str, note_list: List[Note]) -> None:
        """
        Generate enhanced context data based on needs and notes.
        
        Args:
            data_output_base_dir: Directory for output data
            needs_file_name: Filename for the input needs file
            context_enhanced_res_file_name: Filename for the output context-enhanced results
            note_list: A list of Note objects
        """
        all_notes, all_note_str = self._clearn_and_prepare_data_from_memory(note_list=note_list)
        
        # Load the JSON file directly
        with open(data_output_base_dir + "/" + needs_file_name, 'r', encoding='utf-8') as file:
            initial_needs = json.load(file)
        initial_needs = self._extract_initial_needs(initial_needs)
        logging.info(f"Extracted {len(initial_needs)} raw initial needs")
        
        # Randomly sample needs for subsequent processing
        if len(initial_needs) > 5000:
            initial_needs = random.sample(initial_needs, 5000)
            logging.info(f"Randomly sampled 5000 initial needs for further processing")
            
            sampled_needs_path = "../raw_data/backup_0206/sampled_needs.json"
            with open(sampled_needs_path, 'w', encoding='utf-8') as f:
                json.dump(initial_needs, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved sampled 5000 needs to {sampled_needs_path}")
        else:
            logging.info(f"Not enough initial needs to sample 5000, using all {len(initial_needs)} needs")
        
        related_notes_for_needs_file_name = "needsAndRelatedNotes.json"
        # Find related notes and todos for each need
        needsAndRelatedNotesTodos_res = self._find_related_notes_and_todos(
            initial_needs, all_notes, all_note_str, 
            data_output_base_dir + "/" + related_notes_for_needs_file_name
        )
        logging.info(f"Found related notes and todos for {len(needsAndRelatedNotesTodos_res)} needs")
        
        # context enhance
        context_enhanced_needs = self._context_enhance(needsAndRelatedNotesTodos_res)
        logging.info(f"Context enhanced {len(context_enhanced_needs)} needs")
        
        # Combine initial_need, related_note_todos, and context_enhanced_need
        combined_data = []
        for i, item in enumerate(needsAndRelatedNotesTodos_res):
            combined_item = {
                "initial_need": item["initial_need"],
                "related_notes": item["related_notes"],
                "context_enhanced_need": context_enhanced_needs[i]
            }
            combined_data.append(combined_item)
            
        logging.info(f"Combined {len(combined_data)} needs")
        
        # Save the combined data to a JSON file
        output_file_path = data_output_base_dir + "/" + context_enhanced_res_file_name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)


    def expert_response_generator(self, data_output_base_dir: str, context_enhanced_res_file_name: str, output_file_name: str) -> None:
        """
        Generate expert responses for enhanced context data.
        
        Args:
            data_output_base_dir: Directory for output data
            context_enhanced_res_file_name: Filename for the input context-enhanced results
            output_file_name: Filename for the output expert responses
        """
        with open(data_output_base_dir + "/" + context_enhanced_res_file_name, 'r', encoding='utf-8') as f:
            needs = json.load(f)
        
        max_workers = 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(self._process_single_need, needs),
                total=len(needs),
                desc="Processing needs"
            ))

        with open(data_output_base_dir + "/" + output_file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


    def _process_single_need(self, need: Dict) -> Dict:
        """
        Process a single need to generate expert responses.
        
        Args:
            need: Dictionary containing need data
            
        Returns:
            A dictionary with the original need, expert responses, and related notes
        """
        logging.info(f"need: {need['initial_need']}")
        expert_responses = self._get_expert_response(need['initial_need'])
        return {
            "initial_need": need['initial_need'],
            "expert_responses": expert_responses,
            "related_notes": need["related_notes"]
        }


    def _get_expert_response(self, need: str) -> List[str]:
        """
        Get expert responses for a given need.
        
        Args:
            need: The need string to get responses for
            
        Returns:
            A list of expert response strings
        """
        responses = []
        for _ in range(self.multi_time):
            try:
                response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": expert_response_prompt.format(preferred_language=self.preferred_language)},
                            {"role": "user", "content": need}
                        ],
                        temperature=0.8,
                        max_tokens=1000,
                    ).choices[0].message.content
                responses.append(response)
            except Exception as e:
                logging.error(traceback.format_exc())
                continue
        return responses


    def _prompt_generator(self, initial_need: str, expert_response: str, related_notes: List[Dict]) -> List[str]:
        """
        Generate prompts for different types of responses.
        
        Args:
            initial_need: The initial need string
            expert_response: The expert response string
            related_notes: List of related notes
            
        Returns:
            A list of prompt strings
        """
        related_notes_str = ""
        for note in related_notes:
            related_notes_str += self.format_note(note) + "\n"

        prompts = []
        prompts.append(coarse_grained_prompt_a.format(user_name=self.user_name, user_request=initial_need, expert_response=expert_response, global_bio=self.user_bio, preferred_language=self.preferred_language))
        prompts.append(coarse_grained_prompt_b.format(user_name=self.user_name, user_request=initial_need, expert_response=expert_response, global_bio=self.user_bio, preferred_language=self.preferred_language))
        prompts.append(fine_grained_prompt_a.format(user_name=self.user_name, user_request=initial_need, expert_response=expert_response, related_notes=related_notes_str, preferred_language=self.preferred_language))
        prompts.append(fine_grained_prompt_b.format(user_name=self.user_name, user_request=initial_need, expert_response=expert_response, related_notes=related_notes_str, preferred_language=self.preferred_language))
        prompts.append(fine_grained_prompt_c.format(user_name=self.user_name, user_request=initial_need, expert_response=expert_response, related_notes=related_notes_str, preferred_language=self.preferred_language))
        return prompts


    def _generate_all_prompts(self, initial_need: str, expert_responses: List[str], 
                          related_notes: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Generate all prompts for a given need and expert responses.
        
        Args:
            initial_need: The initial need string
            expert_responses: List of expert response strings
            related_notes: List of related notes
            
        Returns:
            A tuple containing a list of all prompt strings and a list of prompt metadata dictionaries
        """
        all_prompts = []
        prompt_metadata = []  # Store metadata for each prompt
        
        for expert_response in expert_responses:
            prompts = self._prompt_generator(initial_need, expert_response, related_notes)
            for i, prompt in enumerate(prompts):
                prompt_type = {
                    0: "coarse_grained_a", 
                    1: "coarse_grained_b",
                    2: "fine_grained_a",
                    3: "fine_grained_b",
                    4: "fine_grained_c",
                }[i]
                
                all_prompts.append(prompt)
                prompt_metadata.append({
                    "related_notes": related_notes,
                    "initial_need": initial_need,
                    "expert_response": expert_response,
                    "prompt_type": prompt_type
                })
        
        return all_prompts, prompt_metadata


    def _process_prompt(self, prompt: str, metadata: Dict, output_file: str) -> None:
        """
        Process a single prompt and save the result to a file.
        
        Args:
            prompt: The prompt string
            metadata: Metadata dictionary for the prompt
            output_file: Path to the output file
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=1000,
                response_format={"type": "json_object"},
            ).choices[0].message.content
            
            result = {
                "related_notes": metadata["related_notes"],
                "initial_need": metadata["initial_need"],
                "expert_response": metadata["expert_response"],
                "response": response,
                "prompt_type": metadata["prompt_type"]
            }
            
            # Write result immediately to JSONL file
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(result)
            logging.info("record saved")
        except Exception as e:
            logging.error(f"Error processing prompt: {e}")


    def _process_prompts_with_threading(self, all_prompts: List[str], prompt_metadata: List[Dict], 
                                    output_file: str, max_workers: int = 10) -> None:
        """
        Process all prompts using multi-threading.
        
        Args:
            all_prompts: List of all prompt strings
            prompt_metadata: List of prompt metadata dictionaries
            output_file: Path to the output file
            max_workers: Maximum number of worker threads
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for prompt, metadata in zip(all_prompts, prompt_metadata):
                future = executor.submit(
                    self._process_prompt,
                    prompt,
                    metadata,
                    output_file
                )
                futures.append(future)
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)


    def gen_context_critic_data(self, data_output_base_dir: str, expert_response_file_name: str, out_file_name: str) -> None:
        """
        Generate context critic data.
        
        Args:
            data_output_base_dir: Directory for output data
            expert_response_file_name: Filename for the input expert responses
            out_file_name: Filename for the output critic data
        """
        if os.path.exists(data_output_base_dir + "/" + out_file_name):
            os.remove(data_output_base_dir + "/" + out_file_name)

        with open(data_output_base_dir + "/" + expert_response_file_name, 'r', encoding='utf-8') as f:
            expert_responses = json.load(f)
        
        total_all = []
        total_all_meta = []

        for item_idx, item in enumerate(tqdm(expert_responses)):
            initial_need = item['initial_need']
            expert_responses = item['expert_responses']
            related_notes = item['related_notes']
            
            # Generate all prompts
            all_prompts, prompt_metadata = self._generate_all_prompts(
                initial_need, 
                expert_responses,
                related_notes,
            )
            
            indices = random.sample(range(len(all_prompts)), 1)
            for idx in indices:
                total_all.append(all_prompts[idx])
                total_all_meta.append(prompt_metadata[idx])
            
        logging.info(f"total_all: {len(total_all)}")

        # Process all prompts using multi-threading
        self._process_prompts_with_threading(
            total_all,
            total_all_meta,
            data_output_base_dir + "/" + out_file_name
        )
        

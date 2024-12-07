import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from collections import Counter
from openai import OpenAI

from tqdm import tqdm

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True,
                max_model_len=40000
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Implements a Tree-of-Thought style approach:
        1. Generate multiple reasoning candidates (thoughts) for each query.
        2. Consolidate these candidates into a single best answer.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Step 1: Extract chunks and embeddings
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        chunk_embeddings = self.calculate_embeddings(chunks)
        query_embeddings = self.calculate_embeddings(queries)

        # Step 2: Retrieve relevant sentences for each query
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            relevant_chunks_mask = chunk_interaction_ids == interaction_id
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Compute cosine similarity and sort by relevance
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)
            retrieval_results = relevant_chunks[(-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]]
            batch_retrieval_results.append(retrieval_results)

        # Step 3: First stage prompt (Divergent Thinking) - Encourage step-by-step reasoning
        # We modify the system prompt to encourage the model to produce reasoned answers.
        # We will generate multiple outputs (n=10).
        formatted_prompts_round1 = self.format_prompts(
            queries, 
            query_times, 
            batch_retrieval_results, 
            system_prompt=(
                "You are provided with a question and various references. "
                "First, think step-by-step about the provided references and the question. "
                "Then provide a reasoned but concise candidate answer. "
                "Do not finalize the answer yet; just provide one candidate reasoning path and answer. "
                "If you are unsure, say 'I don't know'."
            )
        )

        if self.is_server:
            # Server mode (OpenAI API)
            responses_round1 = []
            # Generate multiple reasoning candidates
            for prompt in formatted_prompts_round1:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_name,
                    messages=prompt,
                    n=10,
                    top_p=0.9,
                    temperature=0.7,
                    max_tokens=100,
                )
                # Gather all candidate thoughts
                candidate_answers = [choice.message.content for choice in response.choices]
                responses_round1.append(candidate_answers)
        else:
            # Offline VLLM mode
            responses_round1 = self.llm.generate(
                formatted_prompts_round1,
                vllm.SamplingParams(
                    n=10,
                    top_p=0.9,
                    temperature=0.7,
                    skip_special_tokens=True,
                    max_tokens=100,
                ),
                use_tqdm=False
            )
            # responses_round1 is a list of vllm.GenerationResult
            # Each GenerationResult has multiple outputs
            # Convert them to a list of list of strings
            responses_round1 = [
                [output.text for output in response.outputs] for response in responses_round1
            ]

        # Step 4: Second stage prompt (Convergent Thinking) - Consolidate the candidates
        # Now we have multiple candidate reasoning paths for each query.
        # We'll ask the model to select the best final answer.
        formatted_prompts_round2 = self.format_consolidation_prompts(queries, query_times, batch_retrieval_results, responses_round1)

        if self.is_server:
            final_responses = []
            for prompt in formatted_prompts_round2:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_name,
                    messages=prompt,
                    # Just one best final answer is needed here
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    max_tokens=50,
                )
                final_responses.append(response.choices[0].message.content)
        else:
            final_generation = self.llm.generate(
                formatted_prompts_round2,
                vllm.SamplingParams(
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    skip_special_tokens=True,
                    max_tokens=50,
                ),
                use_tqdm=False
            )
            final_responses = [gen.outputs[0].text for gen in final_generation]

        return final_responses

    def format_prompts(self, queries, query_times, batch_retrieval_results, system_prompt=None):
        """
        Formats the prompts for the first round (ToT reasoning stage).
        If system_prompt is None, it uses the default system prompt.
        """
        if system_prompt is None:
            system_prompt = (
                "You are provided with a question and various references. "
                "Your task is to answer the question succinctly, using the fewest words possible. "
                "If the references do not contain the necessary information to answer the question, "
                "respond with 'I don't know'. There is no need to explain the reasoning."
            )

        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

            user_message += f"{references}\n------\n\n"
            user_message += "Think step-by-step about the references and the question:\n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts

    def format_consolidation_prompts(self, queries, query_times, batch_retrieval_results, all_candidate_answers):
        """
        Format prompts for the second round, where we consolidate multiple candidate answers.
        We'll present the candidate answers from the first round and ask the model to pick the best final answer.
        """
        system_prompt = (
            "You previously generated multiple candidate reasoning paths for the given question. "
            "Now read all the candidate answers and their reasoning. Then select the SINGLE best final answer "
            "that is most likely correct and concise, using the given references. "
            "If unsure, respond with 'I don't know'."
        )

        formatted_prompts = []
        for i, query in enumerate(queries):
            query_time = query_times[i]
            retrieval_results = batch_retrieval_results[i]
            candidates = all_candidate_answers[i]

            references = "# References\n"
            for snippet in retrieval_results:
                references += f"- {snippet.strip()}\n"
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

            user_message = f"{references}\n------\n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            user_message += "\nHere are candidate answers:\n"
            for idx, candidate in enumerate(candidates):
                user_message += f"Candidate {idx+1}: {candidate.strip()}\n\n"

            user_message += (
                "Now, select the single best final answer from these candidates. "
                "If none are satisfactory or certain, respond with 'I don't know'."
            )

            if self.is_server:
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts

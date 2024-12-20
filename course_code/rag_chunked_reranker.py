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
import pickle as pkl

from openai import OpenAI

from tqdm import tqdm

from FlagEmbedding import FlagReranker

import itertools

#### CONFIG PARAMETERS ---

# Define the number of best sentences to pre-filter for with cosine similarity
NUM_SENTENCES_TO_CONSIDER = 50
# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 40
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 200
# Set the minimum length for each context sentence (in characters).
MIN_CONTEXT_SENTENCE_LENGTH = 100
# Set the 
MAX_PARENT_PARAGRAPH_LENGTH = 700
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000
THRESHOLD = 0.1


# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

### CONFIG PARAMETERS END---

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
        text = soup.find("section")
        if text is not None:
            text = text.get_text(" ", strip=True)
        else:
            text = soup.find("article")
            if text is not None:
                text = text.get_text(" ", strip=True)
            else:
                text = soup.find("body")
                if text is not None:
                    text = text.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces
                else:
                    text = soup.get_text(" ", strip=True)

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""], [0]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []
        chunk_parents = []
        counter = 0
        id_counter = 0
        current_sentences = []
        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            if counter != 0 and counter + end - start > MAX_PARENT_PARAGRAPH_LENGTH:
                chunks.extend(current_sentences)
                chunk_parents.extend([id_counter for i in current_sentences])
                current_sentences = []
                id_counter += 1
                counter = 0
            counter += max(end - start, MAX_CONTEXT_SENTENCE_LENGTH)
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            current_sentences.append(sentence)
        chunks.extend(current_sentences)
        chunk_parents.extend([id_counter for i in current_sentences])
            

        return interaction_id, chunks, chunk_parents

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
        chunk_parent_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            value = ray.get(response_ref)  # Blocking call until parallel execution is complete
            interaction_id, _chunks, _chunk_parents = value
            chunk_dictionary[interaction_id].extend(_chunks)
            chunk_parent_dictionary[interaction_id].extend(_chunk_parents)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_parents, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary, chunk_parent_dictionary)

        return chunks, chunk_parents, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary, chunk_parent_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_parents = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = set()
            for ind, chunk in enumerate(_chunks):
                if chunk not in unique_chunks:
                    unique_chunks.add(chunk)
                    chunks.append(chunk)
                    chunk_parents.append(chunk_parent_dictionary[interaction_id][ind])
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        try:
            chunks = np.array(chunks)
            chunk_parents = np.array(chunk_parents)
        except ValueError:
            print(chunks)
            raise Exception
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_parents, chunk_interaction_ids

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None, relevance_scores_path=None, num_context_sentences=NUM_CONTEXT_SENTENCES):
        if not relevance_scores_path:
            self.use_precomputed_scores = False
            self.relevance_scores = None
        else:
            self.use_precomputed_scores = True
            with open(relevance_scores_path, 'rb') as f:
                self.relevance_scores = pkl.load(f)
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()
        self.num_context_sentences = num_context_sentences

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
                gpu_memory_utilization=0.8, # VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True,
                max_model_len=23376
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

        self.reranker_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices=['cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def calculate_rankings(self, query, sentences):
        return self.reranker_model.compute_score(zip(itertools.repeat(query), sentences), normalize=True)

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
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_parents, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        batch_scores = []
        chunk_embeddings = self.calculate_embeddings(chunks)
        query_embeddings = self.calculate_embeddings(queries)


        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            # print("Relevant chunks")
            # print(relevant_chunks)
            relevant_chunks_parent_ids = chunk_parents[relevant_chunks_mask]
            # print("Relevant chunks parent ids")
            # print(relevant_chunks_parent_ids)
            # print("Relevant parents")
            # print(relevant_parents)
            # Calculate cosine similarity between query and chunk embeddings,
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            cosine_filter = (-cosine_scores).argsort()[:NUM_SENTENCES_TO_CONSIDER]
            # and retrieve top-N results.
            cosine_filtered_chunks = relevant_chunks[cosine_filter]
            cosine_filtered_chunk_parent_ids = relevant_chunks_parent_ids[cosine_filter]

            bge_rerankings = self.reranker_model.compute_score(list(zip(itertools.repeat(query), cosine_filtered_chunks)), normalize=True)
            bge_rerankings = np.asarray(bge_rerankings)

            chunks_to_keep = (-bge_rerankings).argsort()[:self.num_context_sentences]
            scores = bge_rerankings[chunks_to_keep]
            retrieval_results = cosine_filtered_chunks[chunks_to_keep]
            # print(retrieval_results)
            retrieval_results_parent_ids = cosine_filtered_chunk_parent_ids[chunks_to_keep]
            
            retrieval_results_parents = []
            retrieval_results_parents_score = []
            retrieval_results_tracker = set()
            retrieval_results_score = defaultdict(int)
            # print(relevant_parents)
            # print(retrieval_results_parent_ids)
            for ind, value in enumerate(retrieval_results_parent_ids):
                # if value not in retrieval_results_tracker:
                retrieval_results_tracker.add(value)
                retrieval_results_score[value] = max(retrieval_results_score[value], scores[ind])
                # else:
                # retrieval_results_score[value] = max(retrieval_results_score[value], scores[ind])
            # print(retrieval_results_tracker)
            for id in retrieval_results_tracker:
                specific_parent_filter = relevant_chunks_parent_ids == id
                retrieval_results_parents.append(" ".join(relevant_chunks[specific_parent_filter]))
                retrieval_results_parents_score.append(retrieval_results_score[id])
            # print(retrieval_results_parents)
            # print(retrieval_results_parents_score)
            # chunks_to_keep = np.where(scores > THRESHOLD)
            # scores = bge_rerankings[chunks_to_keep]
            batch_scores.append(retrieval_results_parents_score)
            batch_retrieval_results.append(retrieval_results_parents)

        # Prepare formatted prompts from the LLM
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results, batch_scores)
        # print(formatted_prompts[0])

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answer = response.outputs[0].text
                answers.append(str(answer).lower().rstrip('.'))

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[], relevance_scores = None):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.

        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        system_prompt = "You are provided with a question and various references in order of relevance. Relevance scores from 0-1 for each are also provided at the end of each reference. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. Do not explain your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            references = ""
            user_message = ""
            # print(retrieval_results)
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    # print(relevance_scores[_idx])
                    # print(_snippet_idx)
                    references += f"- {snippet.strip()}, (score = {round(float(relevance_scores[_idx][_snippet_idx]), 3)})\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
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
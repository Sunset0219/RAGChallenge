import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor


class JinaReranker:
    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()
    

# 使用qwen3的LLM
class LLMReranker:
    def __init__(self):
        self.llm = self.set_up_llm()
        # 从 prompts 模块加载提示词模板
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        
        # 加载用于结构化输出(Structured Output)的 Pydantic 模型定义
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      
    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key="sk-734641e3a00b41d3b2b4e6f6a82c83d3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        return llm
    
    def get_rank_for_single_block(self, query, retrieved_document):
        # 单文档打分函数 
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        
        completion = self.llm.beta.chat.completions.parse(
            model="qwen3-max",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_single_block
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
        
        return response_dict

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        #  将多个文档内容拼接成一个长字符串，用 Block 1, Block 2 区分
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )

        completion = self.llm.beta.chat.completions.parse(
            model="qwen3-max",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_multiple_blocks
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
      
        return response_dict

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        """
        # 负责分批、并发调用、计算加权分。

        # Create batches of documents
        #  将文档列表切分成小批次，例如 [[doc1, doc2, doc3, doc4], [doc5...]]/
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        # 大模型给的评分的权重和向量检索的权重
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # Get ranking for single document
                # 调用单文档打分
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # Calculate combined score - note that distance is inverted since lower is better
                # 计算混合分数: LLM分 * 0.7 + 向量分 * 0.3
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            # 所有文档同时跑，并行
            with ThreadPoolExecutor(max_workers= 1 ) as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            # 批量处理
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                # 调用批量打分
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                # 如果 LLM 返回的结果数量少于输入的文档数量
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    # 打印出哪个文档没拿到分，方便调试
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    # 补齐缺失的结果，给0分，防止程序崩溃
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                # 将分数回填到文档对象中
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            # 并发
            with ThreadPoolExecutor(max_workers= 1 ) as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            #  将二维列表 [[batch1_results], [batch2_results]] 展平为一维列表
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        # 根据混合检索的分数排序
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results

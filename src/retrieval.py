import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
    # 根据公司名称检索，根据文本索引检索
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_path = None
        # 确定公司文档位置
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                 # 找到元数据中公司名匹配的文档
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
                    
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
            
        # Load corresponding BM25 index
        # 加载BM25索引
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # Get the document content and BM25 index
        # 获取文档的切片(chunks)和页面(pages)数据
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # Get BM25 scores for the query
        # 对查询语句分词（简单的空格分词）并获取 BM25 分数
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        #  确定实际返回的数量（防止请求数大于总数）
        actual_top_n = min(top_n, len(scores))
        # 获取分数最高的索引，从高到低排序
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            # 获取对应的文本块
            chunk = chunks[index]
             # 找到该块所属的整页内容
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            # 如果要求返回整页，且该页未被处理过
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        # 使用qwen Embedding模型
        llm = OpenAI(
            api_key="sk-734641e3a00b41d3b2b4e6f6a82c83d3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=None,
            max_retries=2
            )
        return llm
    
    @staticmethod
    def set_up_llm():
        load_dotenv()
        llm = OpenAI(
            api_key="sk-734641e3a00b41d3b2b4e6f6a82c83d3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=None,
            max_retries=2
            )
        return llm

    def _load_dbs(self):
        # 加载向量索引，以及数据库，将 JSON 文档和对应的 .faiss 向量索引文件配对并加载到内存中
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 两个字符串的cos ，计算相似度
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-v4")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        
        target_report = None
        # 在内存中查找相应的报告
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        # 将查询语句转化为向量
        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-v4"
        )
        # 转换为 numpy 数组并重塑形状以适配 FAISS 输入
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # 在向量数据库中query相似的topn，distances 是距离值，indices 是对应的索引ID
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
    
        retrieval_results = []
        seen_pages = set()
        # 遍历搜索结果
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            # 根据索引找回原文切片
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            # 页面去重
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
            
        return retrieval_results
    # 暴力检索，给出所有本公司的页面
    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


class HybridRetriever:
    # 混合检索：向量检索和llm reranking
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: Name of the company to search documents for
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from vector DB
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            
        Returns:
            List of reranked document dictionaries with scores
        """

        # Get initial results from vector retriever
        # 第一步：先使用向量检索获取较大的候选集 
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        # Rerank results using LLM
        # 使用LLM对候选数据集进行重排序
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        # 返回混合结果的topn
        return reranked_results[:top_n]

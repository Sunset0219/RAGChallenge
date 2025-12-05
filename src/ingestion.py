import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        # 从文本块列表创建 BM25 索引
        #  注意：对于中文，这里必须换成 jieba 分词，否则 BM25 无法工作。
        # 但这个比赛是纯英文的，所以 split() 够用了。
        # # jieba.lcut 会把 "今天天气不错" 变成 ['今天', '天气', '不错']
        # tokenized_chunks = [jieba.lcut(chunk) for chunk in chunks]
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.
        
        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        # 处理所有报告并保存独立的 BM25 索引
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        # 遍历每个json
        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            # 对每个文档创建一个索引
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    # 创建向量检索的索引
    def __init__(self):
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        # 使用qwen3
        llm = OpenAI(
            api_key="sk-734641e3a00b41d3b2b4e6f6a82c83d3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=None,
            max_retries=2
        )
        return llm
    # 使用qwen3的Embedding
    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-v4") -> List[float]:
        
        # 空文本检查
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        # 批处理逻辑：OpenAI API 有大小限制，如果列表太长，切成 1024 个一组
        if isinstance(text, list):
            text_chunks = [text[i:i + 10] for i in range(0, len(text), 10)]
        else:
            text_chunks = [text]

        embeddings = []
        for chunk in text_chunks:
            # 调用llm的接口
            # 使用 text-embedding-3-large 模型（维度 3072），效果比 ada-002 好
            response = self.llm.embeddings.create(input=chunk, model=model)
            embeddings.extend([embedding.embedding for embedding in response.data])
        
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        # 将 Python 列表转换为 numpy 数组 (FAISS 要求的格式)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        # 获取向量维度 (text-embedding-3-large 默认是 3072)
        dimension = len(embeddings[0])

        # 创建索引：IndexFlatIP
        # Flat: 暴力搜索，不压缩，精度最高（因为数据量不大，单文件几千个 chunk，暴力搜很快）
        # IP: Inner Product (内积)。因为 OpenAI 的向量是归一化的，所以 内积 = 余弦相似度
        
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        # 将向量加入索引
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        # 单处理每个文档
        #  拿到所有 Chunk 的文本
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        # 调用 API 变成向量
        embeddings = self._get_embeddings(text_chunks)
        # 建立索引
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        # 批处理每个文档
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            # 返回单处理的索引，一个文档一个索引， 
            index = self._process_report(report_data)
            # 保存为 .faiss 文件
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")